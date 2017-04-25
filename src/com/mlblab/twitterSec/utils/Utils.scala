package com.mlblab.twitterSec.utils

import org.apache.log4j.FileAppender
import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.apache.log4j.Priority
import org.apache.log4j.SimpleLayout
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object Utils {
    case class LinkedResult(rank: Int, confidence: Double, actualApk: String, estimatedApk: String)

    def reconstructDataFromLinkedResult(sc: SparkContext, linkdResultsPath: String, linkedDbPath: String, threshold: Double) : (RDD[(String,String)],RDD[LinkedResult]) = {
      val linkedResults = sc.objectFile[LinkedResult](linkdResultsPath)
      val textData = sc.objectFile[(String,(String,String))](linkedDbPath) // apk -> (tweets,appstore)
    
      val confMax = linkedResults.map(_.confidence).max
      
      val toTake = linkedResults.filter(result => result.confidence * (1 / confMax) > threshold)
        
      val (takenCorrect,takenIncorrect,totalCorrect,totalIncorrect) = (toTake.filter(_.rank == 0).count,toTake.filter(_.rank != 0).count,linkedResults.filter(_.rank == 0).count,linkedResults.filter(_.rank != 0).count)
      getLogger.warn(s"taken_correct: $takenCorrect, taken_incorrect: $takenIncorrect, totalCorrect: $totalCorrect, totalIncorrect: $totalIncorrect" +
                s"taken_acc: ${takenCorrect/toTake.count.toDouble}, total_acc: ${totalCorrect/linkedResults.count.toDouble}")
      
      val results = toTake.map(_.estimatedApk -> 1)
            .join(textData)
            .map(x => x._1 -> x._2._2._1)
            
      (results,toTake)
    }
    
    private var logger:Logger = _
    
    def getLogger() = {
      if(logger == null) {
        logger = LogManager.getRootLogger
        val appender = new FileAppender(new SimpleLayout(),"logs/log_" + System.nanoTime() + ".log", false)
        appender.setThreshold(Priority.WARN)
        logger.addAppender(appender)
        logger
      }
      
      logger
    }
}