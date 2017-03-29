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
import com.mlblab.twitterSec.utils.FeatureReductionMethod._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import com.mlblab.twitterSec.utils.ClassifierUtils
import com.mlblab.twitterSec.utils.Utils._

object DecomposedTextualSimiliarity {
  val path = "linkingTextEn.obj"
  val useNGram = true

  var sc: SparkContext = _
  var sql: SQLContext = _
  val log = LogManager.getRootLogger
  
  var data: RDD[(String,(String,String))] = _
  var apks: RDD[String] = _
  
  def main(args: Array[String]) = {
    setup()
    
    if(sc.isLocal)
      data = sc.parallelize(sc.objectFile[(String,(String,String))](path).take(200)).repartition(40).cache // apk -> (tweets,appstore)
    else
      data = sc.objectFile[(String,(String,String))](path).repartition(480).cache // apk -> (tweets,appstore)

    apks = data.map(_._1)
    val ct = apks.count.toInt
    
    val methods = Seq(PCA, SVD) // feature reduction methods
    val components = List(0/*, .25*ct, .5*ct, .75*ct*/).map(_.toInt) // number of components to reduce to with feature reduction
    val ns = Seq(/*2,*/3) // n's to experiment with for nGrams
    val nGramDF = Seq(2/*,3*/) // document count min for nGrams
    val NOGramDF = Seq(3,5,7,10,15) // document count min for single terms
    
    ns.foreach { n => {
      val dfMins = n match {
        case 1 => NOGramDF
        case _ => nGramDF
      }
      
      for(component <- components; /*method <- methods;*/ dfMin <- dfMins) iterate(component, /*method*/SVD, n, dfMin)
    }}
    
    // vary DF and n of n-gram
    //components.foreach { case components => iterate(components.toInt, PCA, 1, 5) }
  }
  
  def iterate(components: Int, reducer: FeatureReductionMethod, n: Int, dfMin: Int) = {
    val docVectors = createTerms(data.flatMap(x => Seq(x._1 + "_t" -> x._2._1, x._1 + "_m" -> x._2._2)), n, dfMin)
    log.warn(s"terms.size: ${docVectors.first._2.size}")    
    val termsMatrix = MathUtils.transposeRowMatrix(new RowMatrix(docVectors.map(_._2)))
    
    val docVectorsIndex = docVectors.zipWithIndex.map(x => x._1._1 -> x._2.toInt).collectAsMap // apk_{t,m} -> column
    val docIndexMap = docVectors.zipWithIndex.map(x => x._2.toInt -> x._1._1).collectAsMap // column -> apk_{t,m}
    var docVectorsReduced = new IndexedRowMatrix(docVectors.map(x => new IndexedRow(docVectorsIndex(x._1),x._2)))
    val boilerplateInfo = s"reducer: $reducer, components: $components, nGram: $n, docFrequencyMin:$dfMin"
    
    if(components != 0) {
      // create reduced space
      val reducedSpace = reducer match {
        case PCA => jointReducedSpacePCA(termsMatrix, components)
        case SVD => jointReducedSpaceSVD(termsMatrix, components)
      }
      
      val reducedSpaceLocal = new DenseMatrix(reducedSpace.rows.count.toInt, reducedSpace.rows.first.size, reducedSpace.rows.flatMap(_.toArray).collect)
      log.warn(s"dims of reducedSpace: ${reducedSpace.numRows}x${reducedSpace.numCols}, $boilerplateInfo")
      log.warn(s"dims of reducedSpaceLocal: ${reducedSpaceLocal.numRows}x${reducedSpaceLocal.numCols}, $boilerplateInfo")
      
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
        val metaRowRanked = twitterRow.vector.toArray.toList
                            .zipWithIndex.filter{ case (value,idx) => docIndexMap(idx).endsWith("_m") } // only rank against other metadata, don't count tweets in ranking
                            .map(_._1).sorted.reverse
        val metaVal = twitterRow.vector.toArray(mCol)
        val bestMatch = docIndexMap(twitterRow.vector.toArray.toList.indexOf(metaRowRanked.head))
        
        LinkedResult(metaRowRanked.indexOf(metaVal),metaVal,apk, bestMatch.substring(0, bestMatch.lastIndexOf('_')))
      }}.collect
    
    val total = apks.count - accum.value

    val perfect = ranks.count(x => x.rank == 0)
    val correctAtLevels = List(.01, .03, .05, .1, .15, .2, .3).map { threshold => threshold -> ranks.filter(x => x.rank <= threshold*total).size }
    
    log.warn(s"dims of docVectorsReduced: ${docVectorsReduced.numRows}x${docVectorsReduced.numCols}, $boilerplateInfo")
    log.warn(s"dims of similarity matrix: ${cosineSimilarityMatrix.numRows}x${cosineSimilarityMatrix.numCols}, $boilerplateInfo")
    
    log.warn(s"total missing: ${accum.value}, $boilerplateInfo")
    
    ////// THINGS
    val (perfConf,nonPerfConf,confAvg) = (MathUtils.mean(ranks.filter(_.rank == 0).map(_.confidence)), MathUtils.mean(ranks.filter(_.rank != 0).map(_.confidence)), MathUtils.mean(ranks.map(_.confidence)))
    val confStdDev = MathUtils.stddev(ranks.map(_.confidence), confAvg)
    log.warn(s"average confidence when right: $perfConf, when wrong: $nonPerfConf, overall confidence average: $confAvg, confidence std dev: $confStdDev")
    
    val metrics = ClassifierUtils.rankedMetrics(sc.parallelize(ranks.map(x => (x.rank,x.confidence))))
    log.warn(s"auPRC: ${metrics.areaUnderPR}, $boilerplateInfo")
    log.warn(s"precision: ${metrics.pr.collect.mkString(", ")}, $boilerplateInfo")
    /////// END THINGS
    
    log.warn(s"for a perfect match, $perfect were recalled, making acc_0: ${perfect/total.toDouble}, $boilerplateInfo")
    // how to reconstruct going forward ... the tweets of actualApk matched with the metadata of estimatedApk
    sc.parallelize(ranks, 1).saveAsObjectFile("results.obj")
    correctAtLevels.foreach{ case (threshold,correct) => log.warn(s"at threshold: $threshold, $correct were recalled, making acc_$threshold: ${correct/total.toDouble}, $boilerplateInfo") }
  }
  
  def jointReducedSpaceSVD(termsMatrix: RowMatrix, components: Int) : RowMatrix = {
    val svd = termsMatrix.computeSVD(components, computeU = true)
    val sMat = DenseMatrix.diag(svd.s)
    
    log.warn(s"dims of U: ${svd.U.numRows}x${svd.U.numCols}")
    log.warn(s"dims of sMat: ${sMat.numRows}x${sMat.numCols}")
    
    svd.U.multiply(sMat)
  }
  
  def jointReducedSpacePCA(termsMatrix: RowMatrix, components: Int) : RowMatrix = {
    val pca = termsMatrix.computePrincipalComponents(components)
    termsMatrix.multiply(pca)
  }

  /**
   * @param texts the (appID -> text) mappings
   * @param n the N to use in N-Gram. Can be 1, which will just skip n-grams
   * @param dfMin document frequency min to use in the CountVectorizer
   */
  def createTerms(texts: RDD[(String,String)], n:Int, dfMin: Int) : RDD[(String,Vector)] = {
    val df = sql.createDataFrame(texts).toDF("appID", "tweetText")
    val linkedTweetsSeperated = new Tokenizer().setInputCol("tweetText").setOutputCol("words").transform(df)
    val linkedTweetsCleaned = (new StopWordsRemover()).setInputCol("words").setOutputCol("filtered").transform(linkedTweetsSeperated)
    import org.apache.spark.sql.functions._
    
    // replacement functions
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

    var cleaned = if (n > 1) {
      val ngram = new NGram().setN(n).setInputCol("filteredMod").setOutputCol("filteredModNGram")
      // Quick Fix for local install of 1.6.0, due to SPARK-12746, make output column nullable
      val ngramTransformed = MathUtils.setNullableStateOfArrayColumn(ngram.transform(linkedTweetsCleanedHtttp), "filteredModNGram", true)
      val cvModel = new CountVectorizer().setMinDF(dfMin).setInputCol("filteredModNGram").setOutputCol("features").fit(ngramTransformed)
      cvModel.transform(ngramTransformed)
    } else {      
      val cvModel = new CountVectorizer().setMinDF(dfMin).setInputCol("filteredMod").setOutputCol("features").fit(linkedTweetsCleanedHtttp)
      cvModel.transform(linkedTweetsCleanedHtttp)
    }
    
    cleaned.select("appID", "features").rdd.map(x => x.getString(0) -> x.getAs[Vector]("features"))
  }
  
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