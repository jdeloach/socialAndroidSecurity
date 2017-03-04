package com.mlblab.twitterSec.collect

import org.apache.spark.SparkConf
import java.io.PrintWriter
import java.io.FileOutputStream
import org.apache.spark.SparkContext
import com.google.gson.Gson
import twitter4j.auth.OAuthAuthorization
import org.apache.spark.streaming.twitter.TwitterUtils
import java.text.SimpleDateFormat
import org.apache.spark.streaming.StreamingContext
import twitter4j.conf.ConfigurationBuilder
import java.io.File
import org.apache.spark.streaming.Seconds
import scala.reflect.io.Directory
import java.util.Date

object EventTweetCollector {
  private var numTweetsCollected = 0L
  private var partNum = 0
  private var gson = new Gson()
  private var twitterKeywords = Array("trump", "sotu", "congress")
  private val outputDirectory = "/Volumes/lacie0/Tweets_TrumpEvent/"
  private var log: PrintWriter = _

  def main(args: Array[String]) {
    val intervalSecs = 60 // seconds to flush
    val partitionsEachInterval = 1
    
    val outputDir = new File(outputDirectory.toString)
    outputDir.mkdirs()
    val formatter = new SimpleDateFormat("yyyy.MM.dd.HH.mm");    
    
    println("Initializing Streaming Spark Context...")
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
      .setMaster("local[2]")
    val sc = new SparkContext(conf)
    val ssc = new StreamingContext(sc, Seconds(intervalSecs))

    val builder = new ConfigurationBuilder()
    builder.setOAuthAccessToken("13556852-il4c9Mmy3Rr5fke3Cc96V4yKiR9kZlv7MeKw4Vu3t")
    builder.setOAuthAccessTokenSecret("3rnZLBhDbkAPnpz5vkr3aBSPFKD9z3wPqPjKHVMOKIUOX")
    builder.setOAuthConsumerKey("mmxbTShtNfuVkT0chdd6A")
    builder.setOAuthConsumerSecret("BHfigeqpzQ4uwO9rRdClDZzW99UeiHMFgt4shtGRKZs")
    val auth = new OAuthAuthorization(builder.build())
    
    val tweetStream = TwitterUtils.createStream(ssc, Some(auth), twitterKeywords)
    val accum = sc.accumulator(0)
    
    tweetStream.foreachRDD((rdd, time) => {
      val count = rdd.count()
      if (count > 0) {

        val outputRDD = rdd
          .map(gson.toJson(_))
        
        outputRDD.repartition(1).saveAsTextFile(outputDirectory + "part_" + formatter.format(new Date()) + ".dist")
        accum.add(1)
        
        // rotate logs every 60 minutes
        if(accum.value >= 60) {
          val files = (new Directory(new File(outputDirectory))).dirs.filter(_.name.startsWith("part_")).toList
          val toMerge = sc.textFile(outputDirectory + "part_*")
          
          toMerge.repartition(1).saveAsTextFile(outputDirectory + "data_" + System.currentTimeMillis() + ".dist")
          files.foreach(_.deleteRecursively)
          accum.setValue(0)
        }
      }
    })
    
    ssc.start()
    ssc.awaitTermination()
    log.close
  }
}