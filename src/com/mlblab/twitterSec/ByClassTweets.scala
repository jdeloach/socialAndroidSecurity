package com.mlblab.twitterSec

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object ByClassTweets {
  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName).setMaster("local[10]")
    val sc = new SparkContext(conf)
        
    //val linkedTweets = sc.sequenceFile[String,String]("/Users/jdeloach/data/linkedTweetsNov14.json").map(x => (x._1,x._2.replace("\n", "").replace("\r", "")))
    //  .groupByKey // group by appID
      
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val df = sqlContext.read.json("/Users/jdeloach/data/linkedTweetsNov14.json")
    val linkedTweets = df.rdd.map(x => DBUtils.appIdFromUrl(x.getString(x.fieldIndex("col"))) -> x.getString(x.fieldIndex("text")))
          .groupByKey
          
    val scannerDB = sc.textFile("/Users/jdeloach/Developer/workspaceML/twitterMLProject/data/appIdsAndScannerCounts.csv")
                      .map{x => val a = x.split(","); (a(0),a(1).toInt)}
                      .collectAsMap
                      
    val benignTweets = linkedTweets.filter{ case (appID,tweets) => scannerDB.contains(appID) && scannerDB(appID) == 0 }
    val malwareTweets = linkedTweets.filter{ case (appID,tweets) => scannerDB.contains(appID) && scannerDB(appID) >= 3 }

    val uniqueByApp = linkedTweets.count
    val inDbBenign = benignTweets.count
    val inDbMalware = malwareTweets.count
    val inDbTotal = inDbBenign + inDbMalware
    
    println(s"uniqueByApp: $uniqueByApp, inDbBenign: $inDbBenign, inDbMalware: $inDbMalware, inDbTotal: $inDbTotal")
    
    /*benignTweets.map(x => x._1 + "," + x._2.mkString(",")).saveAsTextFile("benignTweetsNov14.txt")
    malwareTweets.map(x => x._1 + "," + scannerDB(x._1) + "," + x._2.mkString(",")).saveAsTextFile("malwareTweetsNov14.txt")*/
  }
}