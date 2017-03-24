package com.mlblab.twitterSec.utils

import java.util.ArrayList
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import scala.collection.JavaConversions._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.types.StringType
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

object MathUtils {
  def transposeRowMatrix(m: RowMatrix): RowMatrix = {
    val transposedRowsRDD = m.rows.zipWithIndex.map{case (row, rowIndex) => rowToTransposedTriplet(row, rowIndex)}
      .flatMap(x => x) // now we have triplets (newRowIndex, (newColIndex, value))
      .groupByKey
      .sortByKey().map(_._2) // sort rows and remove row indexes
      .map(buildRow) // restore order of elements in each row and remove column indexes
    new RowMatrix(transposedRowsRDD)
  }

  def transposeIndexedRowMatrix(m: IndexedRowMatrix): IndexedRowMatrix = {
    val transposedRowsRDD = m.rows.map{row => rowToTransposedTriplet(row.vector, row.index)}
      .flatMap(x => x) // now we have triplets (newRowIndex, (newColIndex, value))
      .groupByKey
      .sortByKey() // sort rows and remove row indexes
      .map(x => buildIndexedRow(x._1,x._2)) // restore order of elements in each row and remove column indexes
    new IndexedRowMatrix(transposedRowsRDD)
  }

  def buildIndexedRow(index: Long, rowWithIndexes: Iterable[(Long, Double)]): IndexedRow = {
    val resArr = new Array[Double](rowWithIndexes.size)
    rowWithIndexes.foreach{case (index, value) =>
        resArr(index.toInt) = value
    }
    IndexedRow(index, Vectors.dense(resArr))
  }

  def rowToTransposedTriplet(row: Vector, rowIndex: Long): Array[(Long, (Long, Double))] = {
    val indexedRow = row.toArray.zipWithIndex
    indexedRow.map{case (value, colIndex) => (colIndex.toLong, (rowIndex, value))}
  }

  def buildRow(rowWithIndexes: Iterable[(Long, Double)]): Vector = {
    val resArr = new Array[Double](rowWithIndexes.size)
    rowWithIndexes.foreach{case (index, value) =>
        resArr(index.toInt) = value
    }
    Vectors.dense(resArr)
  }

  implicit def svdExtensions(rm: RowMatrix) = new {
    def computeMultiKSVD(kVals: Traversable[Int], computeU: Boolean = false): Traversable[(Int,SingularValueDecomposition[RowMatrix,Matrix])] = {
      val kMax = kVals.max
      val svd = rm.computeSVD(kMax, computeU)
      // U: MxK, S: KxK, V: NxK
      val m = rm.numRows.toInt
      val n = rm.numCols.toInt

      kVals.map(k => {
        k -> SingularValueDecomposition(reduceMatrix(svd.U, m, k), reduceVector(svd.s,k), Matrices.dense(n, k, svd.V.toArray.slice(0,n*k)))
      })
    }
  }

  def reduceVector(vec: Vector, length: Int) : Vector = Vectors.dense(vec.toArray.slice(0,length))

  def reduceMatrix(mat: RowMatrix, mNew: Int, nNew: Int): RowMatrix = {
    // Potential Bug: What if mat.rows isn't ordered?
    val res = if(mNew < mat.numRows && nNew < mat.numCols) { // reduce columns and rows
      val adjusted = mat.rows
        .zipWithIndex
        .filter(_._2 <= mNew)
        .map { case (vec,idx) => Vectors.dense(vec.toArray.slice(0,mNew)) }
      new RowMatrix(adjusted)
    } else if(mNew < mat.numRows) { // reduce only rows
      new RowMatrix(mat.rows.zipWithIndex.filter(_._2 <= mNew).map(_._1))
    } else { // reduce only columns
      new RowMatrix(mat.rows.zipWithIndex.map { case (vec,idx) => Vectors.dense(vec.toArray.slice(0,nNew)) })
    }

    assert(res.numRows == mNew)
    assert(res.numCols == nNew, s"incorrect col size, should be: $nNew, is ${res.numCols}")

    res
  }

  // Borrowed, with modification, from: https://gist.github.com/softprops/3936429
  def mean(xs: Traversable[Double]): Double = xs match {
    case Nil => 0.0
    case ys => ys.reduceLeft(_ + _) / ys.size.toDouble
  }

  // Borrowed, with modification, from: https://gist.github.com/softprops/3936429
  def stddev(xs: Traversable[Double], avg: Double): Double = xs match {
    case Nil => 0.0
    case ys => math.sqrt((0.0 /: ys) {
     (a,e) => a + math.pow(e - avg, 2.0)
    } / xs.size)
  }

  def toBreeze(vector: Vector) : breeze.linalg.Vector[scala.Double] = vector match {
      case sv: SparseVector => new breeze.linalg.SparseVector[Double](sv.indices, sv.values, sv.size)
      case dv: DenseVector => new breeze.linalg.DenseVector[Double](dv.values)
  }

  def fromBreeze(breezeVector: breeze.linalg.Vector[Double]): Vector = {
    breezeVector match {
      case v: breeze.linalg.DenseVector[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: breeze.linalg.SparseVector[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: breeze.linalg.Vector[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  /**
   * Borrowed with appreciation from: http://stackoverflow.com/a/33195510
   */
  def setNullableStateOfArrayColumn(df: DataFrame, cn: String, nullable: Boolean) : DataFrame = {
    val schema = df.schema
    val newSchema = StructType(schema.map {
      case StructField(c, t, n, m) if c.equals(cn) => StructField(c, ArrayType(StringType, nullable), n, m)
      case y: StructField => y
    })
    df.sqlContext.createDataFrame(df.rdd, newSchema)
  }

  /**
   * If Index = 0, that means the right document was selected, e.g. a correct label/a match.
   * Rank Confidence is, in it's raw form, the cosine similarity between the tweet text and metadata text.
   * We scale this on a 0-1 range as the "score," as higher is more confident in the label.
   */
  def rankedMetrics(ranks: RDD[(Int,Double)]) : BinaryClassificationMetrics = {
    // ranks[(Index, 0=correct,Confidence)]
    val confMax = ranks.map(_._2).max
    val adjusted = ranks.map { case (rank,conf) => val label = rank match { case 0 => 1d case _ => 0d }; val score = conf * (1 / confMax); (score,label) }
    new BinaryClassificationMetrics(adjusted)
  }
}

object FeatureReductionMethod extends Enumeration {
  type FeatureReductionMethod = Value
  val SVD, PCA = Value
}
