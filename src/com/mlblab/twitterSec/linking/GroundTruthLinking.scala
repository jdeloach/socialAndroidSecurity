package com.mlblab.twitterSec.linking

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.linalg.Vector
import com.mlblab.twitterSec.utils.FeaturePrepUtils
import com.mlblab.twitterSec.utils.Utils
import com.mlblab.twitterSec.utils.Utils._
import com.mlblab.twitterSec.utils.VectorizerForm._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.linalg.Vectors
import com.mlblab.twitterSec.utils.MathUtils
import com.mlblab.twitterSec.utils.ClassifierUtils

object GroundTruthLinking {
  
  var sc: SparkContext = _
  var sql: SQLContext = _
  val log = Utils.getLogger
  
  var partitions = 100
  
  var metas: RDD[(String,String)] = _
  var tweets: RDD[(String,String)] = _
  
  // 1. Data Prep (1:N cardinality on links) -- what if it gets one link but not both??
  // 2. Run Various Experiments (Simple, WMF, various parameters, etc.)
  // 3. Return results in READABLE, and PARSEABLE way with ALL (auPRC, auROC, PR points) data!
  
  def experiment(n:Int = 3, dfMin: Int = 2, stem: Boolean = true, numActives: Int = 10) = {
    val docVectorsBase = FeaturePrepUtils.createTerms(sql, metas.map(x => x._1 + "_m" -> x._2) ++ tweets.map(x => x._1 + "_t" -> x._2), n, dfMin, TFIDF, stem)
    val docVectors = docVectorsBase.filter(x => x._2.numActives >= numActives).cache()
    
    /// Require at least 1:1 post-cleaning
    val metasBase = docVectorsBase.filter(_._1.endsWith("_m")).count
    val (mVec, tVec) = (docVectors.filter(_._1.endsWith("_m")), docVectors.filter(_._1.endsWith("_t")))
    val (mKeys, tKeys) = (mVec.map(x => removeTail(x._1)).collect.toSet, tVec.map(x => removeTail(x._1)).collect.toSet)
    val toKeep = mKeys & tKeys
    
    val lost = docVectorsBase.count - docVectors.count
    log.warn(s"total docs lost: $lost, meaning ${toKeep.size} apps have both metas and tweets left, we lost: ${metasBase - mKeys.size} metadatas from their own issues")
    //log.warn(s"REMINDER: WE ARE NOT APPLYING THE 1:1 MIN ON THIS .2 SAMPLE SIZE TEST")
    // End Requirement Cleaning
   
    var dv = docVectors.filter(x => toKeep.contains(removeTail(x._1)))
    var bp = ""
    
    val isDefault = n == 3 && dfMin == 2 && stem == true && numActives == 10
    val kDef = 800; val ks = Seq(2400, 800, 1600, 200); 
    val iterationDef = 5; val iterations = Seq(2, 5, 15, 20);
    val alphaDef = 100; val alphas = Seq(10, 20, 50, 100, 200/*, 400, 800*/);
    val thresholds = Seq(0/*, 0.6, 0.7, 0.8, 0.9*/)
    val addon = if(isDefault) "" else s"n, $n, dfMin, $dfMin, stem: $stem, numActives, $numActives"

    bp = s"dimRed, none"
    val results = cosineSimilarity(docVectors.filter(x => toKeep.contains(removeTail(x._1))), 0)
    evaluateResults(results, bp)
    
    /*for(iteration <- Seq(5, 15, 25); alpha <- Seq(50, 100); lambda <- Seq(0.01, 0.05)) {
      bp = s"dimRed, wmf, k, $kDef, iterations, $iteration, alpha, $alpha, lambda, $lambda"
      dv = wmf(docVectors.filter(x => toKeep.contains(removeTail(x._1))), kDef, iteration, alpha, lambda)      
      val results = cosineSimilarity(dv, 0)
      evaluateResults(results, bp)      
    }*/
    
    if(!isDefault) {
      bp = s"dimRed, wmf, k, $kDef, iterations, $iterationDef, alpha, $alphaDef, $addon"
      dv = wmf(docVectors.filter(x => toKeep.contains(removeTail(x._1))), kDef, iterationDef, alphaDef)      
      val results = cosineSimilarity(dv, 0)
      evaluateResults(results, bp)
    } else {
      /*for(k <- ks) {
        bp = s"dimRed, wmf, k, $k, iterations, $iterationDef, alpha, $alphaDef, $addon"
        dv = wmf(docVectors.filter(x => toKeep.contains(removeTail(x._1))), k, iterationDef, alphaDef)      
        val results = cosineSimilarity(dv, 0)
        evaluateResults(results, bp)  
      }*/
      
      // alpha <- alphas; k <- ks; iteration <- iterations
      /*for(lambda <- Seq(0, 0.01, 0.05, 0.1, 0.2, 0.3)) {
        bp = s"dimRed, wmf, k, $kDef, iterations, $iterationDef, alpha, $alphaDef, LAMBDA: $lambda, $addon"
        dv = wmf(docVectors.filter(x => toKeep.contains(removeTail(x._1))), kDef, iterationDef, alphaDef, lambda)      
        val results = cosineSimilarity(dv, 0)
        evaluateResults(results, bp)  
      }*/
      
      /*for(alpha <- alphas) {
        bp = s"dimRed, wmf, k, $kDef, iterations, $iterationDef, alpha, $alpha, $addon"
        dv = wmf(docVectors.filter(x => toKeep.contains(removeTail(x._1))), kDef, iterationDef, alpha)      
        val results = cosineSimilarity(dv, 0)
        evaluateResults(results, bp)  
      }
     
      for(iteration <-iterations) {
        bp = s"dimRed, wmf, k, $kDef, iterations, $iteration, alpha, $alphaDef, $addon"
        dv = wmf(docVectors.filter(x => toKeep.contains(removeTail(x._1))), kDef, iteration, alphaDef)      
        val results = cosineSimilarity(dv, 0)
        evaluateResults(results, bp)  
      }*/
    }
    
    /*for(threshold <- thresholds) {
      bp = s"dimRed, none, threshold, $threshold"
      val results = cosineSimilarity(docVectors.filter(x => toKeep.contains(removeTail(x._1))), threshold)
        if(results.count > 0)
          evaluateResults(results, bp)
        else
          log.error(s"no results with bp: $bp")
    }*/
  }
  
  def wmf(docVectors: RDD[(String,Vector)], k:Int, iterations: Int, alpha: Int, lambda: Double = 0.1) : RDD[(String,Vector)] = {
    val inverseIndex = docVectors.zipWithIndex().map(x => x._2 -> x._1._1).collectAsMap

    val docRatings = docVectors.zipWithIndex.flatMap{ case ((key,vector),idx) => {
      val rawRatings = vector.toSparse.indices.toList.zip(vector.toSparse.values)
      rawRatings.map { case (wordIndex,value) => Rating(idx.toInt, wordIndex, value.toFloat) }
    }}.cache
    
    val model = ALS.trainImplicit(docRatings, k, iterations, lambda, alpha)
    
    model.userFeatures.map{ case (idx,values) => inverseIndex(idx) -> Vectors.dense(values) }
  }
  
  def cosineSimilarity(docVectors: RDD[(String,Vector)], threshold: Double) : RDD[LinkedResult] = {
    val tweetVectors = docVectors.filter(_._1.endsWith("_t")).mapValues(MathUtils.toBreeze(_))
    val metaVectors = docVectors.filter(_._1.endsWith("_m")).mapValues(MathUtils.toBreeze(_))
    
    val sims = MathUtils.cosineSimilarityBetweenTwoCorpiMulti(metaVectors, tweetVectors, threshold) // (app_m,(app_t,prob))
    //sims.sortBy(_._2.length, false).take(10).foreach(x => log.warn(s"meta: ${x._1} has ${x._2.length} tweets linked"))
    //val sims2 = sims.mapValues(x => if(x.length > 15) x.sortBy(_._2).take(15) else x)
    log.warn(s"num links preLinkedResult, postSims: ${sims.map(_._2.size).sum}, num apps: ${sims.count()}")
    sims.flatMap { case (mId,tweets) => tweets.map{ case (tId,conf) => LinkedResult(idsEqual(mId,tId), conf, mId, tId)} }    
  }
  
  def evaluateResults(links: RDD[LinkedResult], boilerplate: String) = {
    
    //links.map(x => x.estimatedApk -> x).groupByKey.mapValues(_.size).saveAsTextFile("distr_guessed_" + boilerplate.replaceAll(", ", "_") + ".txt")
    //links.map(x => x.actualApk -> x).groupByKey.mapValues(_.size).saveAsTextFile("distr_actual_" + boilerplate.replaceAll(", ", "_") + ".txt")
    
    val metrics = ClassifierUtils.rankedMetrics(links.map(x => (x.rank,x.confidence)))
    val bp = boilerplate.replaceAll(", ", "_")
    val prcOut = "results/exp1/" + sc.applicationId + "_" + boilerplate.replaceAll(", ", "_") + ".prc"
    val rocOut = "results/exp1/" + sc.applicationId + "_" + boilerplate.replaceAll(", ", "_") + ".roc"
    val correct = links.filter(_.rank == 0)
    val incorrect = links.filter(_.rank != 0)
    
    log.warn(s"RESULT, numLinks, ${links.count}, auPRC, ${metrics.areaUnderPR()}, auROC, ${metrics.areaUnderROC()}, BOILERPLATE, $boilerplate")
    log.warn(s"average conf when right: ${correct.map(_.confidence).sum/correct.count.toDouble}, when wrong: ${incorrect.map(_.confidence).sum/incorrect.count.toDouble}, overall avg conf: ${links.map(_.confidence).sum/links.count.toDouble}")
    log.warn(s"For BP: $boilerplate, writing to $prcOut and $rocOut")
    
    val resultsByApp = links.groupBy(_.actualApk)
    val atLeastOneCorrect = resultsByApp.filter(_._2.filter(_.rank == 0).size > 0).count
    log.warn(s"for ${resultsByApp.count} apps, $atLeastOneCorrect apps have at least one correct linked tweet, making %: ${atLeastOneCorrect/resultsByApp.count.toDouble}")
    log.warn(s"overall acc: ${correct.count/incorrect.count.toDouble}, with ${correct.count} correct")
    
    try {
      metrics.pr().repartition(1).saveAsTextFile(prcOut)
      metrics.roc().repartition(1).saveAsTextFile(rocOut)
    } catch {
      case e: Exception => log.error("Caught exception attempting to save ROC/PRC curve. Oh well.")
    }
  }
  
  def idsEqual(x1: String, x2: String) = if(removeTail(x1) == removeTail(x2)) 0 else 1 // 0 means correct
  def removeTail(x: String) = x.substring(0, x.lastIndexOf('_'))

  def setup() = {
    var conf = if(System.getProperty("os.name").contains("OS X")) new SparkConf().setAppName(this.getClass.getSimpleName + "4gb").setMaster("local[10]") else new SparkConf().setAppName(this.getClass.getSimpleName + "4g-lite") 
    conf.set("spark.driver.maxResultSize", "0")
    sc = new SparkContext(conf)
    sql = new SQLContext(sc)
    sc.setCheckpointDir("/tmp")

    // 346,000 unique tweets that map to a little over 4,000 apps
  }
  
  def main(args: Array[String]) = {
    setup()
    
    //for(sample <- Seq(0.99/*, 0.5*/)) {
      val path = "linkingTextEn_sep.obj" // [(String, (String, Iterable[(Long, String)]))]  --- (appID, (Meta, Iterable[(tID,tText)]))
      val data = if(sc.isLocal) sc.objectFile[(String, (String, Iterable[(Long, String)]))](path, partitions)/*.sample(false, 0.2, 90L)*/ else sc.objectFile[(String, (String, Iterable[(Long, String)]))](path)//.sample(false, sample, 11L)//.filter(_._2._2.size >= 2)
      //log.warn(s"now attempting with sample size of: $sample")
      metas = data.mapValues(_._1)
      tweets = data.flatMap(x => if(x._2._2.size >= 2 && false) x._2._2.map(x._1 -> _._2).take(2) else x._2._2.map(x._1 -> _._2))
      log.warn(s"raw count of metas:${metas.count()}, tweets: ${tweets.count}")
      experiment()
      //experiment(1, 5, true)
      //experiment(3, 2, false)
      //experiment(1, 5, true, 10)
      //experiment(3, 2, false, 10)
      //experiment(3, 2, true, 15)
    //}
  }
}