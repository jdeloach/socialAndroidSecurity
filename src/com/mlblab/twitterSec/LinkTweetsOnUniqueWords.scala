package com.mlblab.twitterSec

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.collection.mutable.HashMap

object LinkTweetsOnUniqueWords {
  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
          .setMaster("local[10]") // use 1-core locally. must set in local mode, goes to cluster choice in hadoop/yarn cluster
    val sc = new SparkContext(conf)
    val path = "/Volumes/lacie0/Tweets11/*.dist" 

    val tweets = DBUtils.loadTwitterFile(sc, path)
    val uniqueKeywords = loadUniqueKeywords(sc)
    println(s"unique keywords: ${uniqueKeywords.size}")
    
    //uniqueKeywords.keys.take(1000).foreach(println)
    
    val matches = tweets.filter { x => !prepString(x.getText).filter(uniqueKeywords.contains(_)).isEmpty }
    
    matches.foreach { tweet => {
      val words = tweet.getText.split(" ")
      val matches = words.filter(uniqueKeywords.contains(_))
      
      println(s"for tweet text: ${tweet.getText}, we matched on keywords: ${matches.mkString(",")}")
    }}
    
    
    println(s"number of matches: ${matches.count}")
  }
  
  def loadUniqueKeywords(sc: SparkContext) : Map[String,Int] = {
    val benInfo = sc.textFile("keywords/playdrone.dat").map { x => x.split("\t").toSeq } // title,creator,packagename
    val malInfo = sc.textFile("keywords/malware.dat").map{ x => x.substring(1,x.length-1).split(",").toSeq.tail } // apkFile,packagename,title
    val appInfo = (benInfo ++ malInfo).repartition(40)
    
    appInfo.flatMap(x=>x).flatMap(prepString(_)).map((_,1)).reduceByKey(_ + _).filter(_._2 == 1).map{case (a,b) => a -> b}.collect.toMap
  }
  
  def prepString(s: String) : Array[String] = removeStopWords(cleanString(s))
  def removeStopWords(s: String) : Array[String] = s.toLowerCase.split(" ").filter(!TweetCollector.stopWords.contains(_))
  def cleanString(s: String) = s.replaceAll("[^\\x00-\\x7F]", "").replace("!","")
  def isNumeric(input: String): Boolean = input.forall(_.isDigit)
}