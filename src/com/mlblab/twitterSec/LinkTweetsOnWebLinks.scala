package com.mlblab.twitterSec

import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

object LinkTweetsOnWebLinks {
  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
          .setMaster("local[10]") // use 1-core locally. must set in local mode, goes to cluster choice in hadoop/yarn cluster
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val path = /*"/Users/jdeloach/data/sample.gz" */ "/Volumes/lacie0/Tweets11/data_*.dist" 

    //val tweets = DBUtils.loadTwitterFile(sc, path).cache
    
    return allLinkedJsonFull(sqlContext, path)
      
    val (groundTruthTweets,tweetCount) = DBUtils.getTweetAndApiIdPairs(sqlContext, path)
    groundTruthTweets.cache
    
    //val groundTruthTweets = tweets.filter(_.getURLEntities.exists(_.getExpandedURL.contains("play.google")))
    
    //tweets.foreach { x => println(x.getURLEntities.mkString(",")) }
    
    //val groundTruthTweets = tweets.filter(!_.getURLEntities.isEmpty).foreach(x => println(x.getURLEntities.mkString(","))) 
    
    
    val benPacks = sc.textFile("keywords/playdrone.dat").map { x => x.split("\t").toList } // title,creator,packagename
    val malPacks = sc.textFile("keywords/malware.dat").map{ x => x.substring(1,x.length-1).split(",").toList.tail } // apkFile,packagename,title
    
    val apks = (benPacks ++ malPacks).flatMap(x => x).collect.map(x => x -> 1).toMap
    
    //val groundTruthTweets = tweets.filter(!_.getURLEntities.filter(_.getExpandedURL.contains("play.google.com")).isEmpty)
    /*val tweetAppIds = groundTruthTweets.flatMap(_.getURLEntities.map(_.getExpandedURL)).map(x => x.substring(x.indexOf('=')+1))
    */val doAppsExistInDb = groundTruthTweets.map(_._2).collect.toList.map(apks.contains(_))
    /*
    tweetAppIds.take(10).foreach(println)
    *///println(s"number of ground truth tweets: ${groundTruthTweets.count}, proportion: ${groundTruthTweets.count/tweetCount.toDouble} of total: $tweetCount")   
    //println(s"the number of ground truth tweets that are in the mysql db is: ${doAppsExistInDb.count(_ == true)}")
    
    
    groundTruthTweets.filter(x => apks.contains(x._2)).repartition(16).map(x => x._1 -> x._2).saveAsSequenceFile("/Users/jdeloach/data/linkedTweetsDec20.out")
  }
  
  def allLinkedJsonFull(sc: SQLContext, path: String) = {
    sc.sparkContext.textFile(path).filter(_.contains("play.google")).saveAsTextFile("/Users/jdeloach/data/linkedTweetsDec20_full2.json")
  }
  
  def allLinkedJson(sqlContext: SQLContext, path: String) = {
    val df = sqlContext.read.json(path)
    //df.registerTempTable("tweets")
 
    val df2 = df.where(df("lang").equalTo("en")).select(df("text"),df("favoriteCount"), df("retweetCount"), df("user.followersCount"), df("user.friendsCount"), df("user.favouritesCount"), df("user.statusesCount"), df("user.createdAt"),explode(df("urlEntities.expandedURL")))
    val matches3 = df2.where(df2("col").like("%play.google%"))
    
    matches3.toJSON.repartition(32).saveAsTextFile("/Users/jdeloach/data/linkedTweetsDec20_full.json")
  }
  
  def allLinked(sqlContext: SQLContext, path: String) = {
    val df = sqlContext.read.json(path)
    df.registerTempTable("tweets")
    val matches = sqlContext
      .sql("select text,urlEntities.expandedURL from tweets where lang = 'en'")
      .map(x => (x.getString(0),x.getSeq[String](1)))
      .filter(x => x._2 != null && x._2.length > 0)
      .flatMap{ case (text,urls) => urls.filter(_.contains("play.google"))
      .map(url => (DBUtils.appIdFromUrl(url),text)) } // appID,text
    
    matches.repartition(1).map(x => x._1 -> x._2).saveAsSequenceFile("/Users/jdeloach/data/allLinkedOctEn.out")
  }
}