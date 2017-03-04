package com.mlblab.twitterSec.linking

import scala.reflect.runtime.universe
import org.apache.log4j.FileAppender
import org.apache.log4j.LogManager
import org.apache.log4j.Priority
import org.apache.log4j.SimpleLayout
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.distributed.IndexedRow

object DecomposedTextualSimiliarity {
  val path = "linkingText.obj"

  var sc: SparkContext = _
  var sql: SQLContext = _
  val log = LogManager.getRootLogger
  
  var data: RDD[(String,(String,String))] = _
  var apks: RDD[String] = _
  var termsMatrix: RowMatrix = _
  
  def main(args: Array[String]) = {
    setup()
    
    if(sc.isLocal)
      data = sc.objectFile[(String,(String,String))](path).sample(false, 0.002).repartition(10) // apk -> (tweets,appstore)
    else
      data = sc.objectFile[(String,(String,String))](path).sample(false, .4).repartition(480) // apk -> (tweets,appstore)

    termsMatrix = MathUtils.transposeRowMatrix(new RowMatrix(createTerms(data.map(x => x._1 -> (x._2._1 + " " + x._2._2))).map(_._2)))
    apks = data.map(_._1)
    val ct = apks.count.toInt
    List(0, .25*ct, .5*ct, .75*ct, ct).foreach { components => iterate(components.toInt) }
  }
  
  def iterate(components: Int) = {
    // create matrice of column is the key, row is the vocab
    val docVectors = createTerms(data.flatMap(x => Seq(x._1 + "_t" -> x._2._1, x._1 + "_m" -> x._2._2)))
    val docVectorsIndex = docVectors.zipWithIndex.map(x => x._1._1 -> x._2.toInt).collectAsMap // apk_{t,m} -> column
    val docIndexMap = docVectors.zipWithIndex.map(x => x._2.toInt -> x._1._1).collectAsMap // column -> apk_{t,m}
    var docVectorsReduced = new IndexedRowMatrix(docVectors.map(x => new IndexedRow(docVectorsIndex(x._1),x._2)))
    
    if(components != 0) {
      // create reduced space
      val reducedSpace = jointReducedSpace(components)
      val reducedSpaceLocal = new DenseMatrix(reducedSpace.rows.count.toInt, reducedSpace.rows.first.size, reducedSpace.rows.flatMap(_.toArray).collect)
      log.warn(s"dims of reducedSpace: ${reducedSpace.numRows}x${reducedSpace.numCols}, components: $components")
      log.warn(s"dims of reducedSpaceLocal: ${reducedSpaceLocal.numRows}x${reducedSpaceLocal.numCols}, components: $components")
      
      // transform to reduced space
      docVectorsReduced = docVectorsReduced.multiply(reducedSpaceLocal)
    }
    
    // calculate cosine similarity matrices
    val cosineSimilarityMatrix = MathUtils.transposeRowMatrix(docVectorsReduced.toRowMatrix()).columnSimilarities() // upper triangular nxn matrix
    val accum = sc.accumulator(0)
    
    val ranks = cosineSimilarityMatrix.toIndexedRowMatrix().rows
      .filter(row => docIndexMap(row.index.toInt).endsWith("_t"))
      .map { twitterRow => {
        val keyName = docIndexMap(twitterRow.index.toInt)
        val apk = keyName.substring(0, keyName.lastIndexOf('_'))
        val mCol = docVectorsIndex(apk + "_m")
        val rowRanked = twitterRow.vector.toArray.toList.sorted.reverse
        val metaVal = twitterRow.vector.toArray(mCol)
        
        rowRanked.indexOf(metaVal)
      }}.collect
    
    val total = apks.count - accum.value

    val correctAtLevels = List(.01, .03, .05, .1, .15, .2, .3).map { threshold => threshold -> ranks.filter(x => x <= threshold*total).size }
    
    log.warn(s"dims of docVectorsReduced: ${docVectorsReduced.numRows}x${docVectorsReduced.numCols}, components: $components")
    log.warn(s"dims of similarity matrix: ${cosineSimilarityMatrix.numRows}x${cosineSimilarityMatrix.numCols}, components: $components")
    
    log.warn(s"total missing: ${accum.value}, components: $components")
    correctAtLevels.foreach{ case (threshold,correct) => log.warn(s"at threshold: $threshold, $correct were recalled, making acc_$threshold: ${correct/total.toDouble}, components: $components") }
  }
  
  def jointReducedSpace(components: Int) : RowMatrix = {
    val svd = termsMatrix.computeSVD(components, computeU = true)
    val sMat = DenseMatrix.diag(svd.s)
    
    log.warn(s"dims of U: ${svd.U.numRows}x${svd.U.numCols}")
    log.warn(s"dims of sMat: ${sMat.numRows}x${sMat.numCols}")
    
    svd.U.multiply(sMat)
  }
  
  def createTerms(texts: RDD[(String,String)]) : RDD[(String,Vector)] = {
    val df = sql.createDataFrame(texts).toDF("appID", "tweetText")
    val linkedTweetsSeperated = new Tokenizer().setInputCol("tweetText").setOutputCol("words").transform(df)
    val linkedTweetsCleaned = (new StopWordsRemover()).setInputCol("words").setOutputCol("filtered").transform(linkedTweetsSeperated)
    import org.apache.spark.sql.functions._
    val dropLinks = udf[Seq[String],Seq[String]] (_.filter(!_.startsWith("http")))
    val linkedTweetsCleanedHtttp = linkedTweetsCleaned.withColumn("filteredMod", dropLinks(col("filtered")))

    val cvModel = new CountVectorizer().setInputCol("filteredMod").setOutputCol("features").fit(linkedTweetsCleanedHtttp)
    cvModel.transform(linkedTweetsCleanedHtttp).select("appID", "features").rdd.map(x => x.getString(0) -> x.getAs[Vector]("features"))
  }
  
  def setup() = {
    var conf = if(System.getProperty("os.name").contains("OS X")) new SparkConf().setAppName(this.getClass.getSimpleName + "4gb").setMaster("local[2]") else new SparkConf().setAppName(this.getClass.getSimpleName + "4g") 
    conf.set("spark.driver.maxResultSize", "4g")
    sc = new SparkContext(conf)
    sql = new SQLContext(sc)
    val appender = new FileAppender(new SimpleLayout(),"logs/log_" + System.nanoTime() + ".log", false)
    appender.setThreshold(Priority.WARN)
    log.addAppender(appender)
  }
}