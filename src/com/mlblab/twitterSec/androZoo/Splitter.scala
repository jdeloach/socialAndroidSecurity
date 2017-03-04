package com.mlblab.twitterSec.androZoo

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

object Splitter {
  val path = "/Users/jdeloach/Documents/ML Data/2016/"
  val existingPath = path + "AndroZoo_apks.txt"
  val dbPath = path + "latest.csv"
  
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
          .setMaster("local[10]")
    val sc = new SparkContext(conf)
        
    val md5s = loadExistingMD5s(path + "appIdsAndMd5sAptoide.csv", sc)
    val sha256s = loadExistingSha256s(existingPath, sc)
    val androZooDb = loadDb(dbPath, sc)
    
    val toDownload = androZooDb.filter(x => !sha256s.contains(x._1) && !md5s.contains(x._2))
    val alreadyDownloadedPlayDrone = androZooDb.filter(x => md5s.contains(x._2)).count
    val alreadyDownloadedAndrozoo = androZooDb.filter(x => sha256s.contains(x._1)).count
    
    println(s"toDownload: ${toDownload.count}, alreadyDownloadedPlayDrone: $alreadyDownloadedPlayDrone, alreadyDownloadedAndrozoo: $alreadyDownloadedAndrozoo")
    
    toDownload.map(_._1).repartition(256).saveAsTextFile(path + "toDownload.parts.txt")
  }
  
  def loadExistingSha256s(path: String, sc: SparkContext) : scala.collection.Map[String,Int] = {
    sc.textFile(path).map(_.replace(".apk", "") -> 1).collectAsMap
  }
  
  def loadExistingMD5s(path: String, sc: SparkContext) : scala.collection.Map[String,Int] = {
    sc.textFile(path).map(x => x.split(",")(1) -> 1).collectAsMap
  }
  
  /**
   * Loads the SHA256s and MD5s of all apps in Play Store
   */
  def loadDb(path: String, sc: SparkContext) : RDD[(String,String)]= {
    // sha256,sha1,md5,dex_date,apk_size,pkg_name,vercode,vt_detection,vt_scan_date,dex_size,markets
    
    // sha256/md5
    sc.textFile(path)
      .zipWithIndex
      .filter(_._2 > 0)
      .map { x => {
        val arr = x._1.split(",")
        val year = arr(3).split('-')(0)
        (arr(0),arr(2),arr(10),year)    
      }}
      .filter(x => x._3 == "play.google.com" && x._4 == "2016")
      .map(x => (x._1,x._2))
  }
}