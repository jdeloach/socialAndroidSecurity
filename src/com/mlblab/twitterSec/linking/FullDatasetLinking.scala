package com.mlblab.twitterSec.linking

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import com.mlblab.twitterSec.utils.FeaturePrepUtils
import com.mlblab.twitterSec.utils.MathUtils
import com.mlblab.twitterSec.utils.Utils
import com.mlblab.twitterSec.utils.Utils.LinkedResult
import com.mlblab.twitterSec.utils.VectorizerForm.TFIDF
import com.mlblab.twitterSec.classify.FeatureVectorizer
import com.mlblab.twitterSec.DBUtils
import org.apache.spark.mllib.regression.LabeledPoint
import com.mlblab.twitterSec.utils.ClassifierUtils
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import breeze.linalg.rank
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.linalg.Vectors

/**
 * This class is a two-step approaching to establishing links in the Twitter and App Store Database.
 * The first step is to use our labeled, ground truth dataset to establish confidence expectations.
 * The second step entails using our full dataset of tweets and app metadatas to find similarities, taking those within the established confidence norms.
 * The third step uses those results for classification.
 */
object FullDatasetLinking {
  val path = "linkingTextEn.obj"
  var tweetPath: String = _
  var metadataPath: String = _
  
  var sc: SparkContext = _
  var sql: SQLContext = _
  val log = Utils.getLogger
  val n = 3; val dfMin = 10//2; 
  
  var tSampleSize: Double = _
  var mSampleSize: Double = _
  var ks: Seq[Int] = _
  var alphas: Seq[Int] = _
  var iterations: Seq[Int] = _
  
  /**
   * NOTE/TODO: This is a confidence interval for tweet->metadata, when we are using meta->tweet in real life
   */
  def stepOne() : Double = {
    // 1. Load Labeled Data
    /*val data = sc.objectFile[(String,(String,String))](path).sample(false, sampleSize).repartition(480).cache
    val docVectors = FeaturePrepUtils.createTerms(sql, data.flatMap(x => Seq(x._1 + "_t" -> x._2._1, x._1 + "_m" -> x._2._2)), n, dfMin, TFIDF).zipWithIndex()

    // 2. Execute DecomposedTextualSimilarity on it
    val results = DecomposedTextualSimiliarity.wtmfInternal(docVectors, k, iterations, n, dfMin, alpha)
    log.warn("num results: " + results._1.size)
    log.warn(results._2)
    DecomposedTextualSimiliarity.evaluateResults(results._1, results._2, sc)
    
    // 3. Using the LinkedResults, establish where our thresholded confidence should be, an optimal point on the auPRC curve if you will.
    val accuracyGoal = 0.9d
    val sortedList = results._1.sortBy(_.confidence)
    def acc(res: Array[LinkedResult]) : Double = { res.count(_.rank == 0) / res.length }
    var accTmp = 0d
    var toKeep = 20 // give us a starting point
    
    while (accTmp > accuracyGoal) {
      accTmp = acc(sortedList.slice(0, toKeep))
      toKeep = toKeep + 1
    }
    log.warn(s"took $toKeep, to achieve an accuracy of $accTmp, targeting: $accuracyGoal")
    log.warn(s"that means we should take confidences as low as: ${sortedList(toKeep).confidence}")
    */
    // 4. Yield that number.
    //sortedList(toKeep).confidence
    .30d
  }
  
  /**
   * Yields (metadataKey,tweetKey,confidence)
   */
  def stepTwo(threshold: Double, k:Int = 200, iterations:Int = 5, alpha:Int = 100, useMulti: Boolean = true) : RDD[(String,Seq[(String,Double)])] = {
    // 1. Load all Tweet Data, Load All Metadata's
    val tweets = sql.read.json(tweetPath).sample(false, tSampleSize).select("id", "text").rdd.filter(!_.isNullAt(0)).map(x => (x.getLong(0) + "_t", x.getString(1)))
    val descriptions = sc.textFile(metadataPath).sample(false, mSampleSize)
                         .map(_.split(",", 2))
                         .filter(_.length == 2)
                         .zipWithIndex
                         .map{x => (x._2,x._1(0),x._1(1).replaceAll("\\<[^>]*>",""))} // NUM, appId, Description
    
      // 1.1. Load labeled data as well, and use that to evaluate accuracy?
    
    // 2. Vectorize all documents
    val docVectors = FeaturePrepUtils.createTerms(sql, tweets ++ descriptions.map(x => (x._2 + "_m") -> x._3), n, dfMin, TFIDF).filter(x => x._2.numActives >= 10).repartition(500) // require at least 5 favlid terms
    
    log.warn(s"linking vocabulary.size: ${docVectors.first._2.size}")

    // 3. Do a Fast-Cosine Similarity
    //--val tweetVectors = docVectors.filter(_._1.endsWith("_t"))/*.repartition(400).persist(StorageLevel.MEMORY_ONLY_2)*/
    //--val metaVectors = docVectors.filter(_._1.endsWith("_m"))
    //val cosSim = MathUtils.cosineSimilarityBetweenTwoCorpi(metaVectors, tweetVectors)
    
    // 4. Establish Linked Results
    //val bestGuess = cosSim.map { case (metaKey,results) => metaKey -> results.toList.sortBy(_._2).reverse.head }
    
    val vecIndex = docVectors.map(_._1).zipWithIndex.collectAsMap
    val inverseIndex = vecIndex.map(x => x._2 -> x._1)
    val docRatings = docVectors.flatMap{ case (key,vector) => {
      val rawRatings = vector.toSparse.indices.toList.zip(vector.toSparse.values)
      rawRatings.map { case (wordIndex,value) => Rating(vecIndex(key).toInt, wordIndex, value.toFloat) }
    }}
    
    val model = ALS.trainImplicit(docRatings, k, iterations, 0.05, alpha)
    val completedDocVectors = model.userFeatures.map{ case (idx,values) => inverseIndex(idx) -> Vectors.dense(values) }
    val tweetVectors = completedDocVectors.filter(_._1.endsWith("_t")).mapValues(MathUtils.toBreeze(_)).persist(StorageLevel.MEMORY_AND_DISK)
    val metaVectors = completedDocVectors.filter(_._1.endsWith("_m")).mapValues(MathUtils.toBreeze(_)).persist(StorageLevel.MEMORY_AND_DISK)
    
    val bestGuess = if(useMulti) MathUtils.cosineSimilarityBetweenTwoCorpiMulti(metaVectors, tweetVectors, threshold) else MathUtils.cosineSimilarityBetweenTwoCorpi2(metaVectors, tweetVectors, threshold)
    
    // 5. Only yield those above the threshold
    log.warn(s"best guess.size: ${bestGuess.count}")
    bestGuess
    //bestGuess.filter(_._3 /*2._2*/ > threshold).map(x => (x._1,(x._2,x._3)))
  }
  
  def stepThree(links: RDD[(String,Seq[(String,Double)])], boilerPlate: String) = {
    log.warn(s"links: ${links.map(x => s"appID: ${x._1}, matchCount:${x._2.length} (tweetIDs,conf):${x._2.map(x => "(" + x._1 + "," + x._2 + ")").mkString(", ")}").collect.mkString("\n")}")
    
    // 1. Load Tweets, Apps that are linked
    val tweetsToLoad = links.flatMap(_._2.map(_._1)).map(x => x.substring(0,x.lastIndexOf('_'))).collect
    val appsToLoad = links.map(_._1).map(x => x.substring(0,x.lastIndexOf('_'))).collect
    val tweetsRaw = FeatureVectorizer.twitterMetricsForIds(sql, tweetPath, tweetsToLoad)
    val appsRaw = DBUtils.loadSelectData2(sc, appsToLoad)
    
    // 2. Establish a single, and joint feature vector
    val linkBase = links.zipWithUniqueId()
    val apps = linkBase.map(x => x._1._1.substring(0,x._1._1.lastIndexOf('_')) -> x._2).join(appsRaw).map(x => x._2._1 -> x._2._2)
    val tweets = linkBase.flatMap(x => x._1._2.map(y => y._1.substring(0,y._1.lastIndexOf('_')).toLong -> x._2)).join(tweetsRaw).map(x => x._2._1 -> x._2._2).groupByKey
    val joint = apps.join(tweets).map{ case (x,(lp,v)) => new LabeledPoint(lp.label, ClassifierUtils.combine(lp.features.toSparse, FeatureVectorizer.reducer("average", v))) }
    
    // 3. Run Classifiers, evaluate performance
    val (noTweetsFolds,withTweetsFolds) = (MLUtils.kFold(apps.values, 5, 412), MLUtils.kFold(joint, 5, 412))
    
    log.warn("apps.size: " + apps.count + ", tweets.size: " + tweets.count + ", joint.size: " + joint.count)
    log.warn("raw pos count: " + joint.filter(_.label == 1).count)
    withTweetsFolds.foreach(fold => log.warn("train pos: " + fold._1.filter(_.label == 1).count + ", test pos:" + fold._2.filter(_.label == 1).count))
    
    val nbPRC = (ClassifierUtils.naiveBayes(noTweetsFolds), ClassifierUtils.naiveBayes(withTweetsFolds))
    val svmPRC = (ClassifierUtils.svm(noTweetsFolds), ClassifierUtils.svm(withTweetsFolds))
    val lrPRC = (ClassifierUtils.logisticRegression(noTweetsFolds), ClassifierUtils.logisticRegression(withTweetsFolds))
    //val rfPRC = (ClassifierUtils.randomForrest(noTweetsFolds), ClassifierUtils.randomForrest(withTweetsFolds))
    
    log.warn(s"feature vector size noTweets: ${apps.first._2.features.size}, withTweets: ${joint.first.features.size}, label: $boilerPlate\n")    
    log.warn(s"instance size: ${apps.count}, pos: ${apps.values.filter(_.label == 1).count}, neg: ${apps.values.filter(_.label == 0).count}, label: $boilerPlate\n")
    log.warn(s"NaiveBayes -- noTweets: ${nbPRC._1}, withTweets: ${nbPRC._2}, label: $boilerPlate\n")
    log.warn(s"SVM -- noTweets: ${svmPRC._1}, withTweets: ${svmPRC._2}, label: $boilerPlate\n")
    log.warn(s"Logistic Regression -- noTweets: ${lrPRC._1}, withTweets: ${lrPRC._2}, label: $boilerPlate\n")
    //log.warn(s"Random Forrest -- noTweets: ${rfPRC._1}, withTweets: ${rfPRC._2} label: $boilerPlate\n")
  }
  
  def setup() = {
    var conf = if(System.getProperty("os.name").contains("OS X")) new SparkConf().setAppName(this.getClass.getSimpleName + "local").setMaster("local[10]") else new SparkConf().setAppName(this.getClass.getSimpleName + "cluster") 
    sc = new SparkContext(conf)
    sc.setCheckpointDir("/tmp")
    sql = new SQLContext(sc)    
    
    tweetPath = if(sc.isLocal) "/Volumes/SAMSUNG/allTweets.txt.gz/part00" else "allTweets.txt.gz"
    metadataPath = if(sc.isLocal) "/Users/jdeloach/Developer/workspaceML/twitterMLProject/data/appIdsAndDescriptions.csv" else "appIdsAndDescriptions.csv"
    tSampleSize = if(sc.isLocal) 0.125 else 0.2d
    mSampleSize = if(sc.isLocal) 0.025 else 0.5d  
    ks = if(sc.isLocal) Seq(200) else Seq(/*200, 400,*/ 800, 1600)
    alphas = if(sc.isLocal) Seq(100) else Seq(/*100, 200,*/ 400)
    iterations = if(sc.isLocal) Seq(5) else Seq(5, 20)
  }

  def main(args: Array[String]) = {
    setup()
    val threshold = stepOne()
    
    //for(iteration <- iterations; alpha <- alphas; useMulti <- Seq(true/*, false*/); k <- ks) {
    val k = 800; val iteration = 5; val alpha = 50; val useMulti = true;
    val links = stepTwo(threshold, k, iteration, alpha, useMulti)
    log.warn(s"number of links: ${links.count}, with a threshold of: $threshold")
    stepThree(links, s"k=$k, alpha=$alpha, iterations=$iteration, useMulti=$useMulti")
    //}
  }
  
  def removeTail(x: String) = x.substring(0, x.lastIndexOf('_'))
}