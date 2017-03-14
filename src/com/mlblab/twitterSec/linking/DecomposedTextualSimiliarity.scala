package com.mlblab.twitterSec.linking

import scala.reflect.runtime.universe
import org.apache.log4j.FileAppender
import org.apache.log4j.LogManager
import org.apache.log4j.Priority
import org.apache.log4j.SimpleLayout
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf
import com.mlblab.twitterSec.utils.MathUtils
import org.apache.spark.mllib.linalg.Vectors

object DecomposedTextualSimiliarity {
  val path = "linkingTextEn.obj"

  var sc: SparkContext = _
  var sql: SQLContext = _
  val log = LogManager.getRootLogger
  
  var data: RDD[(String,(String,String))] = _
  var apks: RDD[String] = _
  var termsMatrix: RowMatrix = _
  var docVectors: RDD[(String,Vector)] = _
  
  def main(args: Array[String]) = {
    setup()
    
    if(sc.isLocal)
      data = sc.parallelize(sc.objectFile[(String,(String,String))](path).take(500)).repartition(40) // apk -> (tweets,appstore)
    else
      data = sc.objectFile[(String,(String,String))](path).sample(false, .5).repartition(480) // apk -> (tweets,appstore)

    docVectors = createTerms(data.flatMap(x => Seq(x._1 + "_t" -> x._2._1, x._1 + "_m" -> x._2._2)))
    log.warn(s"terms.size: ${docVectors.first._2.size}")
    
    termsMatrix = MathUtils.transposeRowMatrix(new RowMatrix(docVectors.map(_._2)))
    apks = data.map(_._1)
    val ct = apks.count.toInt
    List(0, .25*ct, .5*ct, .75*ct, ct).foreach { components => iterate(components.toInt) }
  }
  
  def iterate(components: Int) = {
    // create matrice of column is the key, row is the vocab
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
    
    // replacement funcitons
    val dropLinks = udf[Seq[String],Seq[String]] (_.map(x => if(x.startsWith("http")) "<url>" else x))
    val replaceUserNames = udf[Seq[String],Seq[String]] (_.map(x => if(x.startsWith("@")) "<username>" else x))
    val dropNumbers = udf[Seq[String],Seq[String]] (_.filter(!_.forall(_.isDigit)))
    val removeHashtagSymbol = udf[Seq[String],Seq[String]] (_.map(_.replaceAll("#", "")))
    val removeNonAscii = udf[Seq[String],Seq[String]] (_.map(_.replaceAll("[^\\x00-\\x7F]", "")))
    val removeNonEnglish = udf[Seq[String],Seq[String]] (_.map(_.replaceAll("[^\\p{L}\\p{Nd}]+", "")))

    val linkedTweetsCleanedHtttp = linkedTweetsCleaned.withColumn("filteredMod", 
        removeNonEnglish(
        removeNonAscii(
        removeHashtagSymbol(
        dropNumbers(
        replaceUserNames(
        dropLinks(col("filtered"))))))))

    val cvModel = new CountVectorizer().setInputCol("filteredMod").setOutputCol("features").fit(linkedTweetsCleanedHtttp)
    val cleaned = cvModel.transform(linkedTweetsCleanedHtttp)

    val colsToKeep = cleaned
      .select("appID", "features")
      .map(x => binarizeVector(x.getAs[Vector]("features")))
      .reduce(binarySumVector(_,_))
      .toArray.toList
      .zipWithIndex
      .filter{ case (value,idx) => value > 5 }
      .map(_._2).toArray
        
    val slicer = new VectorSlicer().setInputCol("features").setOutputCol("features_final")
    slicer.setIndices(colsToKeep)
    val dfOutput = slicer.transform(cvModel.transform(cleaned))
    dfOutput.select("appID", "features_final").rdd.map(x => x.getString(0) -> x.getAs[Vector]("features_final"))
  }
  
  def binarySumVector(v1: Vector, v2: Vector) = MathUtils.fromBreeze(MathUtils.toBreeze(v1) + MathUtils.toBreeze(v2))
  def binarizeVector(v: Vector) = Vectors.sparse(v.size, v.toSparse.indices, Array.fill(v.toSparse.indices.size)(1))
  
  def setup() = {
    var conf = if(System.getProperty("os.name").contains("OS X")) new SparkConf().setAppName(this.getClass.getSimpleName + "4gb").setMaster("local[2]") else new SparkConf().setAppName(this.getClass.getSimpleName + "4g-lite") 
    conf.set("spark.driver.maxResultSize", "4g")
    sc = new SparkContext(conf)
    sql = new SQLContext(sc)
    val appender = new FileAppender(new SimpleLayout(),"logs/log_" + System.nanoTime() + ".log", false)
    appender.setThreshold(Priority.WARN)
    log.addAppender(appender)
  }
}