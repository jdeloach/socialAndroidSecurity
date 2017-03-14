package com.mlblab.twitterSec

import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date

import scala.collection.mutable.HashMap
import scala.reflect.io.Directory

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.twitter.TwitterUtils

import com.google.gson.Gson
import com.mlblab.twitterSec.utils.ConfigValues

import twitter4j.auth.OAuthAuthorization
import twitter4j.conf.ConfigurationBuilder

/**
 * Collect at least the specified number of tweets into json text files.
 */
object TweetCollector {
  private var numTweetsCollected = 0L
  private var partNum = 0
  private var gson = new Gson()
  private var twitterKeywords = Array("android", "app", "malware", "mobile")
  val stopWords = Array("a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours  ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves").map { x => x -> 1 }.toMap
  private val outputDirectory = "/Volumes/lacie0/Tweets11/"
  private var log: PrintWriter = _

  def main(args: Array[String]) {
    val numTweetsToCollect = 500000
    val intervalSecs = 60 // seconds to flush
    val partitionsEachInterval = 1
    
    val outputDir = new File(outputDirectory.toString)
    /*(if (outputDir.exists()) {
      System.err.println("ERROR - %s already exists: delete or specify another directory".format(
        outputDirectory))
      System.exit(1)
    }
    outputDir.mkdirs()*/
    log = new PrintWriter(new FileOutputStream(outputDirectory + "log.log", true))
    val formatter = new SimpleDateFormat("yyyy.MM.dd.HH.mm");    
    
    println("Initializing Streaming Spark Context...")
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
      .setMaster("local[10]")
    val sc = new SparkContext(conf)
    val ssc = new StreamingContext(sc, Seconds(intervalSecs))

    val builder = new ConfigurationBuilder()
    builder.setOAuthAccessToken(ConfigValues.TwitterOAuthAccessToken)
    builder.setOAuthAccessTokenSecret(ConfigValues.TwitterOAuthAccessTokenSecret)
    builder.setOAuthConsumerKey(ConfigValues.TwitterOAuthConsumerKey)
    builder.setOAuthConsumerSecret(ConfigValues.TwitterOAuthConsumerSecret)
    val auth = new OAuthAuthorization(builder.build())
    
    val keywords = sc.broadcast(loadKeywordsMap(sc).filter{case (a,b) => a.length > 2 && !stopWords.contains(a)}) // keys must be 3 chars or longer
    diagnostic("finished constructing keywords map")
    val tweetStream = TwitterUtils.createStream(ssc, Some(auth), twitterKeywords)
    val accum = sc.accumulator(0)
    
    tweetStream.foreachRDD((rdd, time) => {
      val count = rdd.count()
      if (count > 0) {

        val outputRDD = rdd
          .filter{ x => tweetContainsKeyword(x.getText.toLowerCase, keywords.value) || x.getURLEntities.filter(_.getExpandedURL.contains("play.google")).size > 0 }
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

    // what if instead of ensuring we consider ex-order equality? e.g. all wor
    
    ssc.start()
    ssc.awaitTermination()
    log.close
  }
  
  /**
   * Returns all in-order sub-word sets in string
   */
  def substringPermutations(words: Seq[String]) : List[Seq[String]] = {
    words.inits.flatMap { x => x.tails.toList.init }.toList
  }
  
  def tweetContainsKeyword(text: String, db: Map[String, Int]) : Boolean = {
    val cleaned = text.split(" ").filter(!stopWords.contains(_)).toSeq
    val subsets = substringPermutations(cleaned).filter(_.size <= 3) // only 3-grams and below are considered for matching
    val matches = subsets.map(_.mkString(" ")).filter(db.contains(_))
    
    if(!matches.isEmpty && Math.random() < .1) { // only execute on 1/10th of tweets
      diagnostic(s"matched tweet $text on keywords ${matches.mkString(";")}")
    }
    
    !matches.isEmpty
  }
  
  def loadKeywordsMap(sc: SparkContext) : Map[String,Int] = {
    val benInfo = sc.textFile("keywords/playdrone.dat").map { x => x.split("\t").toSeq } // title,creator,packagename
    val malInfo = sc.textFile("keywords/malware.dat").map{ x => x.substring(1,x.length-1).split(",").toSeq.tail } // apkFile,packagename,title
    val appInfo = (benInfo ++ malInfo).repartition(40)
    
    appInfo.filter { x => !x.contains("unfound") }.flatMap { phrases => {
      val map = new HashMap[String,Int]
      phrases.foreach { phrase => map.put(phrase.split(" ").filter(!stopWords.contains(_)).mkString(" ").toLowerCase, 1) }
      map.toMap
    }}.groupBy(x => x._1).collect.map(_._1 -> 1).toMap
  }
  
  def diagnostic(line: String) = {
    log.append(new SimpleDateFormat("MM/dd/yyyy HH:mm:ss").format(new Date()) + " " + line + "\n")
  }
}