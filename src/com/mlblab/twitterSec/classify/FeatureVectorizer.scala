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
import breeze.stats.median
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.feature.Normalizer
import com.mlblab.twitterSec.utils.Utils
import com.mlblab.twitterSec.DBUtils
import com.mlblab.twitterSec.utils.FeaturePrepUtils
import org.apache.log4j.LogManager

object FeatureVectorizer {
  val log = Utils.getLogger
  
  /**
   * @param vectorReducerMethod options: {median, head, sum}
   */
  case class FeatureVectorizerProperties(useText: Boolean, n: Int, minDF: Int, vectorReducerMethod: String)
  
  var properties:HashMap[String,Boolean] = HashMap("useAuthorMetrics" -> true, "useEntityMetrics" -> true, "useConversationStatus" -> true, "useText" -> true)
  var df:DataFrame = _
  
  def createVectors(sql: SQLContext, properties: FeatureVectorizerProperties) : RDD[(Utils.LinkedResult,SparseVector)] = {
    /*df = sql.read.json("data/linkedDec20.json")//.sample(false, 0.1)
    val df2 = df.select(df("id"), explode(df("urlEntities.expandedURL")))
    val df3 = df2.where((df2("col")).like("%play.google%"))
    val df4 = df.where(df("id").isin(df3.select(df3("id")).map(_.getLong(0)).collect:_*)).repartition(1000).cache
    df4.write.json("df4.json")
    */
    val df = sql.read.json("data/metrics_raw.json")

    val base = Seq(df("urlEntities"))
    val authorMetrics = Seq(df("followersCount"), df("friendsCount"), df("favouritesCount"), df("statusesCount"))
    val entityMetrics = Seq(df("size(mediaEntities)"), df("size(urlEntities)"))
    val conversationMetrics = Seq(df("(inReplyToUserId > -1)"))
    //val text = Seq(df("text"))
    
    // metrics element
    val metrics = df.select((base ++ authorMetrics ++ entityMetrics ++ conversationMetrics):_*)
      .map(row => appIdFromStatus(row) -> createVector(row))
      .groupByKey
      .map(x => x._1 -> reducer(properties.vectorReducerMethod, x._2))
    
    if(properties.useText) {
      // text part
      //val textualData = df4.select(df("urlEntities"),df("text")).map(row => appIdFromStatus(row) -> row.getString(row.schema.fieldIndex("text"))).cache
      val dataBase = "/Users/jdeloach/Dropbox/Research/MLB Lab/Linking Tweets And Apps/Data/"
      val (textualData,linkedResults) = Utils.reconstructDataFromLinkedResult(sql.sparkContext, dataBase + "Linked Tweets Results/results.obj", dataBase + "Linked Tweets/linkingText.obj", .10)
      log.warn(s"textualData:size: ${textualData.count}")
      val terms = FeaturePrepUtils.createTerms(sql, textualData, properties.n, properties.minDF)
      //val transformedTerms = transformTerms(terms, pcaCount).repartition(500)
      
      //terms.mapValues(_.toSparse)
      // merge feature vectors
      
      // we need to use a MislabeledLabeledPoint hurrrrrr
      val nMetrics = linkedResults.map(res => res.actualApk -> res).join(metrics).map(x => x._2._1 -> x._2._2)
      val termsWithResultKey = linkedResults.map(res => res.actualApk -> res).join(terms).map(x => x._2._1 -> x._2._2.toSparse)
      log.warn(s"num terms: ${terms.count}, num metrics: ${metrics.count}, overlap: ${terms.join(metrics).count} nMetricsOverlap: ${termsWithResultKey.join(nMetrics).count}")
      nMetrics//.join(terms).map{ case (key,vectors) => key -> combine(vectors._2.toSparse,vectors._1) }
    }
    else
    {
      null//metrics
    }
  }
  
  def createVector(status: Row) = {
    Vectors.dense((1 until status.length).map(idx => status.schema.fields(idx).dataType match {
      case DataTypes.BooleanType => if(status.getBoolean(idx)) 1d else 0d
      case DataTypes.IntegerType => status.getInt(idx).toDouble
      case DataTypes.LongType => status.getLong(idx).toDouble
    }).toArray).toSparse
  }
  
  def reducer(reducer: String, list: Iterable[SparseVector]) : SparseVector = {
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
}