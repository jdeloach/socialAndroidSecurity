package com.mlblab.twitterSec

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import com.mlblab.twitterSec.classify.FeatureVectorizer
import java.io.PrintWriter
import java.io.File
import org.apache.log4j.LogManager
import org.apache.log4j.SimpleLayout
import org.apache.log4j.FileAppender
import org.apache.log4j.Priority
import com.mlblab.twitterSec.utils.ClassifierUtils

object TweetAugmentedClassifier {
  var sc: SparkContext = _
  val log = LogManager.getRootLogger
  
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
          //.setMaster("local[10]")
          //.set("spark.driver.memory", "16g")
          //.set("spark.executor.memory", "3g")
          
    sc = new SparkContext(conf)
    val appender = new FileAppender(new SimpleLayout(),"logs/log_" + System.nanoTime() + ".log", false)
    appender.setThreshold(Priority.WARN)
    log.addAppender(appender)

    var twitterVectors:RDD[(String,SparseVector)] = null
    
    Set(/*"head", "sum", */"average"/*, "median"*/).foreach(reducer => {
      List(/*5,*/ 50, 100, 200, 300, 400, 500/*, 1000, 5000, 10000, 0*/).reverse.foreach(pcaCount => {
        twitterVectors = FeatureVectorizer.createVectors(new SQLContext(sc), reducer, pcaCount)
        val headSize = twitterVectors.first._2.size
        assert(twitterVectors.filter(_._2.size != headSize).isEmpty, "Feature vector size is not consistent")
        run(twitterVectors, s"reducer: $reducer, pcaCount: $pcaCount")
      })
    })
  }
  
  def run(twitterVectors: RDD[(String,SparseVector)], label: String) = {
    val validApks = twitterVectors.map(_._1).collect.map(x => x -> 1).toMap
    log.warn(s"number of linked tweets in db ${twitterVectors.count}\n")
    log.warn(s"number of UNIQUE apps tweeted about: ${validApks.size}\n")
    
    val apkVectors = DBUtils.loadSelectData2(sc, validApks.map(_._1).toSeq).cache
    
    log.warn(s"size of apkVectors:${apkVectors.count}\n")
    
    val overlap = (validApks.keySet & apkVectors.keys.collect.toSet).map { x => x -> 1 }.toMap
    val apkSubset = apkVectors.filter(x => overlap.contains(x._1))
    val tweetSubset = twitterVectors.filter(x => overlap.contains(x._1))
    val t = apkSubset.groupWith(tweetSubset).map{ case (appId,(labeledPoints,tweetVectors)) => (appId,labeledPoints.head,tweetVectors.head) }.repartition(40).cache
    //t.saveAsObjectFile(s"t_$label.out")    
    
    val noTweets = t.map(x => (x._2))
    val withTweets = t.map(x => new LabeledPoint(x._2.label, combine(x._2.features.toSparse, x._3.toSparse)))
    
    log.warn(s"number of features in noTweets: ${noTweets.first().features.size} withTweets: ${withTweets.first().features.size}")
    
    val useUndersampling = false
    val noTweetsFolds = MLUtils.kFold(noTweets, 5, 412).map(x => (if(useUndersampling) ClassifierUtils.undersample(x._1.repartition(1000)).cache else x._1.repartition(1000).cache,x._2.repartition(1000).cache))
    val withTweetsFolds = MLUtils.kFold(withTweets, 5, 412).map(x => (if(useUndersampling) ClassifierUtils.undersample(x._1.repartition(1000)).cache else x._1.repartition(1000).cache,x._2.repartition(1000).cache))
    
    val nbPRC = (ClassifierUtils.naiveBayes(noTweetsFolds), ClassifierUtils.naiveBayes(withTweetsFolds))
    val svmPRC = (ClassifierUtils.svm(noTweetsFolds), ClassifierUtils.svm(withTweetsFolds))
    val lrPRC = (ClassifierUtils.logisticRegression(noTweetsFolds), ClassifierUtils.logisticRegression(withTweetsFolds))
    //val rfPRC = (randomForrest(noTweetsFolds), randomForrest(withTweetsFolds))
    
    log.warn(s"feature vector size noTweets: ${noTweets.first.features.size}, withTweets: ${withTweets.first.features.size}, label: $label\n")    
    log.warn(s"instance size: ${noTweets.count}, pos: ${noTweets.filter(_.label == 1).count}, neg: ${noTweets.filter(_.label == 0).count}, label: $label\n")
    log.warn(s"NaiveBayes -- noTweets: ${nbPRC._1}, withTweets: ${nbPRC._2}, label: $label\n")
    log.warn(s"SVM -- noTweets: ${svmPRC._1}, withTweets: ${svmPRC._2}, label: $label\n")
    log.warn(s"Logistic Regression -- noTweets: ${lrPRC._1}, withTweets: ${lrPRC._2}, label: $label\n")
    //output.println(s"Random Forrest -- noTweets: ${rfPRC._1}, withTweets: ${rfPRC._2}\n")
  }
  
  /*def preprocessTweetStats(sc: SparkContext) : RDD[(String,Vector)] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    // text,favoriteCount,retweetCount,followersCount,friendsCount,favouritesCount,statusesCount,createdAt,col
    // not using text,createdAt
    val df = sqlContext.read.json("/Users/jdeloach/data/linkedTweetsNov14.json")
    
    df.rdd.map(x => DBUtils.appIdFromUrl(x.getString(x.fieldIndex("col"))) -> 
                              Array(/*SentimentAnalysisUtils.detectSentimentScore(x.getString(x.fieldIndex("text")))+2, */x.getLong(x.fieldIndex("favoriteCount")).toDouble,x.getLong(x.fieldIndex("retweetCount")).toDouble,
                                   x.getLong(x.fieldIndex("followersCount")).toDouble, x.getLong(x.fieldIndex("friendsCount")).toDouble, 
                                   x.getLong(x.fieldIndex("favouritesCount")).toDouble,x.getLong(x.fieldIndex("statusesCount")).toDouble))
          .groupByKey
          .map(x => x._1 -> Vectors.dense(x._2.map(y => breeze.linalg.Vector(y)).reduce(_ + _).map(z => z / x._2.size).toArray))
  }
  
  def preprocessTweets(sc: SparkContext) : RDD[(String,Vector)] = {
    val linkedTweets = sc.sequenceFile[String,String]("/Users/jdeloach/data/allLinkedOct30En.out")
      .groupByKey // group by appID
      .map(x => (x._1,x._2.mkString(" ").replace("\n", ""))) // and combine tweets on the same app
    
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val df = sqlContext.createDataFrame(linkedTweets).toDF("appID", "tweetText")
    val linkedTweetsSeperated = new Tokenizer().setInputCol("tweetText").setOutputCol("words").transform(df)
    val linkedTweetsCleaned = (new StopWordsRemover()).setInputCol("words").setOutputCol("filtered").transform(linkedTweetsSeperated)/*.select("appID", "filtered").rdd.map(x => (x.getString(0), x.getSeq[String](1).filter(!_.contains("http")))) 
    val (appIds,tweetText) = (linkedTweetsCleaned.map(_._1),linkedTweetsCleaned.map(_._2))

    val hashingTF = new HashingTF()
    val tf = hashingTF.transform(tweetText).cache
    val idf = new IDF(minDocFreq = 2).fit(tf)
    val tfidf = appIds.zip(idf.transform(tf))
    
    tfidf*/

    //val t = new SQLContext(sc).createDataFrame(linkedTweetsCleaned.map(x => Row(x.getString(0), x.getSeq[String](2).filter(!_.startsWith("http"))))
    
    import org.apache.spark.sql.functions._
    val dropLinks = udf[Seq[String],Seq[String]] (_.filter(!_.startsWith("http")))
    val linkedTweetsCleanedHtttp = linkedTweetsCleaned.withColumn("filteredMod", dropLinks(col("filtered")))
    
    val cvModel = new CountVectorizer().setInputCol("filteredMod").setOutputCol("features").fit(linkedTweetsCleanedHtttp)
    val res = cvModel.transform(linkedTweetsCleanedHtttp).select("appID", "features").rdd.map(x => (x.getString(0), x.getAs[Vector]("features")))
    
    res
  }*/
  
  def combine(v1:SparseVector, v2:SparseVector): SparseVector = {
    val size = v1.size + v2.size
    val maxIndex = v1.size
    val indices = v1.indices ++ v2.indices.map(e => e + maxIndex)
    val values = v1.values ++ v2.values
    new SparseVector(size, indices, values)
  }
}