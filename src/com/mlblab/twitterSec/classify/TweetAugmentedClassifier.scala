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
import com.mlblab.twitterSec.classify.FeatureVectorizer._
import java.io.PrintWriter
import java.io.File
import org.apache.log4j.LogManager
import org.apache.log4j.SimpleLayout
import org.apache.log4j.FileAppender
import org.apache.log4j.Priority
import com.mlblab.twitterSec.utils.ClassifierUtils
import com.mlblab.twitterSec.utils.Utils._
import com.mlblab.twitterSec.utils.MislabeledLabeledPoint
import com.mlblab.twitterSec.utils.Utils

object TweetAugmentedClassifier {
  var sc: SparkContext = _
  val log = Utils.getLogger
  
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
          .setMaster("local[10]")
          .set("spark.driver.memory", "24g")
          .set("spark.executor.memory", "16g")
          
    sc = new SparkContext(conf)
    val appender = new FileAppender(new SimpleLayout(),"logs/log_" + System.nanoTime() + ".log", false)
    appender.setThreshold(Priority.WARN)
    log.addAppender(appender)

    var twitterVectors:RDD[(String/*LinkedResult*/,SparseVector)] = null
    
    Set(/*"head", "sum", */"average"/*, "median"*/).foreach(reducer => {
      twitterVectors = FeatureVectorizer.createVectors(new SQLContext(sc), FeatureVectorizerProperties(useText = true, n = 3, minDF = 2, vectorReducerMethod = reducer))
      val headSize = twitterVectors.first._2.size
      assert(twitterVectors.filter(_._2.size != headSize).isEmpty, "Feature vector size is not consistent")
      run(twitterVectors, s"reducer: $reducer")
    })
  }
  
  def run(twitterVectors: RDD[(/*LinkedResult*/String,SparseVector)], label: String) = {
    log.warn(s"number of linked tweets in db ${twitterVectors.count}\n")
    
    val t = twitterVectors.join(DBUtils.loadSelectData2(twitterVectors.context, twitterVectors.keys.collect())).map(x => x._1 -> (x._2._2,x._2._1))
    //val t = constructMislinkedDataset(twitterVectors)
    
    
    val noTweets:RDD[LabeledPoint] = t.map(x => (x._2._1))
    val withTweets:RDD[LabeledPoint] = t.map(x => new LabeledPoint(x._2._1.label, combine(x._2._1.features.toSparse, x._2._2)))
    
    log.warn(s"number of features in noTweets: ${noTweets.first().features.size} withTweets: ${withTweets.first().features.size}")
    log.warn(s"num of vecs: ${t.count}"/*", number of mislinked vecs that got a corresponding wrong label: ${t.filter(x => x._2._1.realLabel != x._2._1.label).count}"*/)
    
    val useUndersampling = false
    val noTweetsFolds = MLUtils.kFold(noTweets, 5, 412).map(x => (if(useUndersampling) ClassifierUtils.undersample(x._1.repartition(1000)).cache else x._1.repartition(1000).cache,x._2.repartition(1000).cache))
    val withTweetsFolds = MLUtils.kFold(withTweets, 5, 412).map(x => (if(useUndersampling) ClassifierUtils.undersample(x._1.repartition(1000)).cache else x._1.repartition(1000).cache,x._2.repartition(1000).cache))
    
    val nbPRC = (ClassifierUtils.naiveBayes(noTweetsFolds), ClassifierUtils.naiveBayes(withTweetsFolds))
    val svmPRC = (ClassifierUtils.svm(noTweetsFolds), ClassifierUtils.svm(withTweetsFolds))
    val lrPRC = (ClassifierUtils.logisticRegression(noTweetsFolds), ClassifierUtils.logisticRegression(withTweetsFolds))
    //val rfPRC = (ClassifierUtils.randomForrest(noTweetsFolds), ClassifierUtils.randomForrest(withTweetsFolds))
    
    log.warn(s"feature vector size noTweets: ${noTweets.first.features.size}, withTweets: ${withTweets.first.features.size}, label: $label\n")    
    log.warn(s"instance size: ${noTweets.count}, pos: ${noTweets.filter(_.label == 1).count}, neg: ${noTweets.filter(_.label == 0).count}, label: $label\n")
    log.warn(s"NaiveBayes -- noTweets: ${nbPRC._1}, withTweets: ${nbPRC._2}, label: $label\n")
    log.warn(s"SVM -- noTweets: ${svmPRC._1}, withTweets: ${svmPRC._2}, label: $label\n")
    log.warn(s"Logistic Regression -- noTweets: ${lrPRC._1}, withTweets: ${lrPRC._2}, label: $label\n")
    //log.warn(s"Random Forrest -- noTweets: ${rfPRC._1}, withTweets: ${rfPRC._2}\n")
  }
  
  /**
   * Takes the estimated twitter data and matches it with the actual binary data. If it impairs it impairs, if it help it helps.
   */
  def constructMislinkedDataset(twitterVectors: RDD[(LinkedResult,SparseVector)]) : RDD[(String,(LabeledPoint,SparseVector))] = {
    val allApks = twitterVectors.flatMap(res => Seq(res._1.actualApk, res._1.estimatedApk)).collect.distinct
    val binaryFeatures = DBUtils.loadSelectData2(sc, allApks) // Apk -> LabeledPoint
    
    twitterVectors
            .map(x => x._1.estimatedApk -> x._2)
            .join(binaryFeatures)
            .map(x => x._1 -> (x._2._2, x._2._1))
  }
}