package com.mlblab.twitterSec

import java.io.File
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.twitter.TwitterUtils
import com.google.gson.Gson
import twitter4j.auth.OAuthAuthorization
import twitter4j.conf.ConfigurationBuilder
import java.io.FileOutputStream
import java.io.PrintWriter

/**
 * Collect at least the specified number of tweets into json text files.
 */
object Politics {
  private var numTweetsCollected = 0L
  private var partNum = 0
  private var gson = new Gson()
  private var twitterKeywords = Array("trump", "clinton")

  def main(args: Array[String]) {
    val outputDirectory = "/Users/jdeloach/tmp/sent_ptx21/"
    val numTweetsToCollect = 500000
    val intervalSecs = 10 // seconds to flush
    val partitionsEachInterval = 1
    
    val outputDir = new File(outputDirectory.toString)
    if (outputDir.exists()) {
      System.err.println("ERROR - %s already exists: delete or specify another directory".format(
        outputDirectory))
      System.exit(1)
    }
    outputDir.mkdirs()

    println("Initializing Streaming Spark Context...")
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
      .setMaster("local[9]")
    val sc = new SparkContext(conf)
    val ssc = new StreamingContext(sc, Seconds(intervalSecs))

    val builder = new ConfigurationBuilder()
    val auth = new OAuthAuthorization(builder.build())
    
    val tweetStream = TwitterUtils.createStream(ssc, Some(auth), twitterKeywords).map(_.getText)
    
    tweetStream.foreachRDD((rdd, time) => {
      val count = rdd.count()
      if (count > 0) {
        val sentimentAverages = rdd.sample(false, .5, 11L).map { x => if(x.toLowerCase().contains("trump")) ("trump",SentimentAnalysisUtils.detectSentiment(x)) else ("clinton",SentimentAnalysisUtils.detectSentiment(x)) }
          .groupByKey
          .map { x => {
            (x._1,time.milliseconds.toString,x._2.map { x => x match {
              case SentimentAnalysisUtils.NOT_UNDERSTOOD => 0
              case SentimentAnalysisUtils.VERY_NEGATIVE => -2
              case SentimentAnalysisUtils.NEGATIVE => -1
              case SentimentAnalysisUtils.NEUTRAL => 0
              case SentimentAnalysisUtils.POSITIVE => 1
              case SentimentAnalysisUtils.VERY_POSITIVE => 2 
            }}.sum.toDouble / x._2.size, x._2.size)
          }}.repartition(partitionsEachInterval)

        sentimentAverages.saveAsTextFile(outputDirectory + "/sentiments_" + time.milliseconds.toString)
      }
    })

    ssc.start()
    ssc.awaitTermination()
  }
}