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

object MathUtils {
  def transposeRowMatrix(m: RowMatrix): RowMatrix = {
    val transposedRowsRDD = m.rows.zipWithIndex.map{case (row, rowIndex) => rowToTransposedTriplet(row, rowIndex)}
      .flatMap(x => x) // now we have triplets (newRowIndex, (newColIndex, value))
      .groupByKey
      .sortByKey().map(_._2) // sort rows and remove row indexes
      .map(buildRow) // restore order of elements in each row and remove column indexes
    new RowMatrix(transposedRowsRDD)
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
  
  def dropColsFromVector(vector: SparseVector, colsToDrop: Array[Int]) : Vector = {    
    val (toKeepSparse,toKeepThis) = vector.indices.toList.zipWithIndex.filter{ case (sparseIndex,thisIndex) => !colsToDrop.contains(sparseIndex) }.unzip
    val toKeepAdjusted = toKeepSparse.map(x => x - colsToDrop.filter(_ < x).size)
    val valuesToKeep = vector.values.toList.zipWithIndex.filter{ case (value,thisIndex) => toKeepThis.contains(thisIndex) }.map(_._1)
    assert(toKeepSparse.length == valuesToKeep.length) // could fail if 0's aren't in values to keep ??
    Vectors.sparse(vector.size - colsToDrop.size,toKeepAdjusted.zip(valuesToKeep))
  }
}