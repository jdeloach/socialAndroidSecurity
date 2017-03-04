package com.mlblab.twitterSec.utils

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.classification.LRLogisticRegressionWithLBFGS

object ClassifierUtils {
  def naiveBayes(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) : Double = {
    val naiveBayesPRC = folds.map{ case (train,test) => {
      
      val model = NaiveBayes.train(train)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x.label) }
      (new BinaryClassificationMetrics(predsAndLabel, 100).areaUnderPR)
    }}
    
    naiveBayesPRC.sum / folds.size.toDouble
  }

  def svm(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) : Double = {
    val naiveBayesPRC = folds.map{ case (train,test) => {
      
      val model = SVMWithSGD.train(train, 100)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x.label) }
      (new BinaryClassificationMetrics(predsAndLabel, 100).areaUnderPR)
    }}
    
    naiveBayesPRC.sum / folds.size.toDouble
  }
  
  def logisticRegression(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) : Double = {
    val naiveBayesPRC = folds.map{ case (train,test) => {
      
      val model = new LogisticRegressionWithLBFGS().run(train)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x.label) }
      (new BinaryClassificationMetrics(predsAndLabel, 100).areaUnderPR)
    }}
    
    naiveBayesPRC.sum / folds.size.toDouble
  }
    
  def randomForrest(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) : Double = {
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 15 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val naiveBayesPRC = folds.map{ case (train,test) => {
      val model = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x.label) }
      (new BinaryClassificationMetrics(predsAndLabel, 100).areaUnderPR)
    }}
    
    naiveBayesPRC.sum / folds.size.toDouble
  }
  
  def lrlr(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])], pTilde: Double, lambdaU: Double) : Double = {
    folds.map{ case (train,test) => {
      val algo = new LRLogisticRegressionWithLBFGS(pTilde, lambdaU).run(train)
      val predsAndLabel = test.map{ x=> (algo.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      (new BinaryClassificationMetrics(predsAndLabel, 100).areaUnderPR)
    }}.sum / folds.size.toDouble
  }
  
  def undersample(training: RDD[LabeledPoint]) : RDD[LabeledPoint] = {
    val positiveClass = training.filter(_.label == 1)
    val negativeClass = training.filter(_.label == 0)
    training.context.makeRDD(negativeClass.takeSample(false, positiveClass.count.toInt, 11L)) ++ positiveClass
  }
}