package com.mlblab.twitterSec

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import twitter4j.Status
import twitter4j.TwitterObjectFactory
import org.apache.spark.mllib.feature.Word2Vec
import scala.collection.mutable.HashMap

object LinkTweets {
  
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
          //.setMaster("local[10]") // use 1-core locally. must set in local mode, goes to cluster choice in hadoop/yarn cluster
    val sc = new SparkContext(conf)
    val path = "parts.file.gz"
    
    val tweets = loadDistributedTweetFile(sc, path).sample(false, .001, 11L)
    val appInfo = sc.textFile("apps_text.txt").map { x => x.split("        ").toSeq }.repartition(40)

    // cache contains map
    //appInfo.foreach { seq => seq.foreach { term => appInfoWords.put(term, 1) } }
    
    
    val appWords = appInfo.flatMap { line => {
      val words = line.flatMap { x => x.split(" ") }
      val map = new HashMap[String,Int]
      words.foreach { word => map.put(word, 1) }
      map.toMap
    }}.groupBy(x => x._1).collect.map(_._1 -> 1).toMap
    
    println(s"Tweets Count: ${tweets.count}")
    println(s"App Info Count: ${appInfo.count}")
    
    // tfidf it, then take the top words and attempt to match
    val tfidfTweets = tweetsToTFIDF(tweets)
    val keywordsForTweets = getKeywords(tfidfTweets)

    val appInfo2 = appInfo.collect
    val lines = keywordsForTweets.map(tweetData => {
      val results = tweetData.flatMap { term => {
        if(appWords.contains(term)) {
          Some(appInfo2.filter{ parts => {
            val words = parts.flatMap{ _.split(" ") }
            words.filter { x => x.toLowerCase() == term.toLowerCase() }.size > 0
          }}.map { x => x.head })
        } else
          None
      }}
      
      s"For Tweet: ${tweetData.mkString(",")}, apps relevant: ${results.flatMap(x=> x).mkString(",")}"
    })
    
    lines.saveAsTextFile(s"tweetMatchingResults_${System.currentTimeMillis}.txt")
    
    // attempt to start matching
    /*val word2vec = new Word2Vec()
    val model = word2vec.fit(appInfo.map { x => x.flatMap { _.split(" ") } })
    
    // foreach tweet
    println(s"Keywords For Tweets Length: ${keywordsForTweets.count}")
    keywordsForTweets.foreach(tweetData => {
      println(s"Keywords Length: ${tweetData.size}")
      
      // foreach word
      tweetData.map { term => {
        // check if word exists first
        if(appWords.contains(term)) {   
          val synonyms = model.findSynonyms(term, 3).map(_._1)
           
          println(s"For word ... $term, here are three synonyms: ${synonyms.mkString(", ")}")
        }
         // now search for synonyms in description
      }}
    })*/
  }
  
  def loadDistributedTweetFile(sc: SparkContext, path: String) : RDD[Status] = {
    sc.textFile(path).map { x => TwitterObjectFactory.createStatus(x) }
  }
  
  /**
   * Returns a tweet (tokenized into terms/words) and the vector of the terms 
   * TF-IDF value.
   */
  def tweetsToTFIDF(tweets : RDD[Status]) : RDD[(Seq[String],Vector)] = {
    val tokenizedTweets = tweets.map { _.getText.split(" ").toSeq }
    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(tokenizedTweets)
    tf.cache()
    val idf = new IDF().fit(tf)
    tokenizedTweets.zip(idf.transform(tf))
  }

  def getKeywords(tfidfTweets: RDD[(Seq[String],Vector)]): RDD[(Seq[(String)])] = {
    tfidfTweets.map{ case (terms,vector) => {
      val zipped = terms.zip(vector.toArray)
      zipped.sortBy(x => x._2).take((terms.size * .2).toInt).map(_._1)
    }}
  }
}

