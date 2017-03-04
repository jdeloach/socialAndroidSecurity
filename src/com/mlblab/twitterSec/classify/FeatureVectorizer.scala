package com.mlblab.twitterSec.classify

import scala.collection.mutable.HashMap
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import com.mlblab.twitterSec.DBUtils
import breeze.stats.median
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.feature.Normalizer

object FeatureVectorizer {
  var properties:HashMap[String,Boolean] = HashMap("useAuthorMetrics" -> true, "useEntityMetrics" -> true, "useConversationStatus" -> true, "useText" -> true)
  var reducer = "median" // median, head, sum
  var df:DataFrame = _
  
  def createVectors(sql: SQLContext, red: String, pcaCount: Int) : RDD[(String,SparseVector)] = {
    reducer = red
    df = sql.read.json("data/linkedDec20.json").sample(false, 0.1)
    val df2 = df.select(df("id"), explode(df("urlEntities.expandedURL")))
    val df3 = df2.where((df2("col")).like("%play.google%"))
    val df4 = df.where(df("id").isin(df3.select(df3("id")).map(_.getLong(0)).collect:_*)).repartition(1000).cache
    
    val base = Seq(df("urlEntities"))
    val authorMetrics = Seq(df("user.followersCount"), df("user.friendsCount"), df("user.favouritesCount"), df("user.statusesCount"))
    val entityMetrics = Seq(size(df("mediaEntities")), size(df("urlEntities")))
    val conversationMetrics = Seq(df("inReplyToUserId") > -1)
    val text = Seq(df("text"))
    
    // metrics element
    val metrics = df4.select((base ++ authorMetrics ++ entityMetrics ++ conversationMetrics):_*)
      .map(row => appIdFromStatus(row) -> createVector(row))
      .groupByKey
      .map(x => x._1 -> reducer(x._2))
    
    if(properties("useText") && pcaCount != 0) {
      // text part
      val textualData = df4.select(df("urlEntities"),df("text")).map(row => appIdFromStatus(row) -> row.getString(row.schema.fieldIndex("text"))).cache
      val terms = createTerms(sql, textualData)
      val transformedTerms = transformTerms(terms, pcaCount).repartition(500)
      
      // merge feature vectors
      transformedTerms/*.join(metrics).map{ case (key,vectors) => key -> combine(vectors._1,vectors._2) }*/
    }
    else
    {
      metrics
    }
  }
  
  def createVector(status: Row) = {
    Vectors.dense((1 until status.length).map(idx => status.schema.fields(idx).dataType match {
      case DataTypes.BooleanType => if(status.getBoolean(idx)) 1d else 0d
      case DataTypes.IntegerType => status.getInt(idx).toDouble
      case DataTypes.LongType => status.getLong(idx).toDouble
    }).toArray).toSparse
  }
  
  def reducer(list: Iterable[SparseVector]) : SparseVector = {
    val merged = reducer match {
      case "head" => list.head
      case "average" => Vectors.dense(list.map(y => breeze.linalg.Vector(y.toArray)).reduce(_ + _).map(z => z / list.size).toArray).toSparse
      case "median" => Vectors.dense(list
              .flatMap(row => row.toArray.toList.zipWithIndex.map(x => x._2 -> x._1)) // emit (col,val)
              .groupBy(_._1) // group by col
              .map(x => median(breeze.linalg.DenseVector(x._2.map(_._2).toArray))) // take the median of the val, by col
              .toArray).toSparse
      case "sum" => Vectors.dense(list.map(y => breeze.linalg.Vector(y.toArray)).reduce(_ + _).toArray).toSparse
      case _ => throw new Exception("Invalid Reducer Option: " + reducer)
    }
    
    combine(merged, Vectors.dense(list.size).toSparse)
  }
  
  // Helper Functions
  def appIdFromStatus(status: Row) = DBUtils.appIdFromUrl(status.getSeq(status.fieldIndex("urlEntities")).toString.split(",")(2))
  
  def combine(v1:SparseVector, v2:SparseVector): SparseVector = {
    val size = v1.size + v2.size
    val maxIndex = v1.size
    val indices = v1.indices ++ v2.indices.map(e => e + maxIndex)
    val values = v1.values ++ v2.values
    new SparseVector(size, indices, values)
  }

  def createTerms(sqlContext: SQLContext, texts: RDD[(String,String)]) : RDD[(String,Vector)] = {
    val df = sqlContext.createDataFrame(texts).toDF("appID", "tweetText")
    val linkedTweetsSeperated = new Tokenizer().setInputCol("tweetText").setOutputCol("words").transform(df)
    val linkedTweetsCleaned = (new StopWordsRemover()).setInputCol("words").setOutputCol("filtered").transform(linkedTweetsSeperated)
    import org.apache.spark.sql.functions._
    val dropLinks = udf[Seq[String],Seq[String]] (_.filter(!_.startsWith("http")))
    val linkedTweetsCleanedHtttp = linkedTweetsCleaned.withColumn("filteredMod", dropLinks(col("filtered")))

    val cvModel = new CountVectorizer().setInputCol("filteredMod").setOutputCol("features").fit(linkedTweetsCleanedHtttp)
    cvModel.transform(linkedTweetsCleanedHtttp).select("appID", "features").rdd.map(x => (x.getString(0), x.getAs[Vector]("features")))
  }
  
  def transformTerms(data: RDD[(String,Vector)], components: Int) : RDD[(String,SparseVector)] = {
    val mat = new RowMatrix(data.map(_._2)) // consider transposing this ... rows might need to be vocab
    
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
    
    val svd = transposeRowMatrix(mat).computeSVD(components, computeU = true)
    val reducedSpace = svd.U.multiply(new DenseMatrix(1, svd.s.size, svd.s.toArray)).rows
    
    val projection = data.map(_._1).zip(reducedSpace).map(x => x._1 -> x._2.toSparse)
    
    val colMins = projection.flatMap(x => List.fromArray(x._2.toArray).zipWithIndex.map(y => y._2 -> y._1))
            .groupBy(_._1)
            .map(x => x._1 -> { val min = x._2.map(_._2).min; if (min < 0) Math.abs(min) else 0 })
            .collectAsMap
    
    val updated = projection.mapValues { x => Vectors.dense(x.toArray.toList.zipWithIndex.map{ case (value,idx) => value + colMins(idx) }.toArray).toSparse }
            
    updated
  }
}