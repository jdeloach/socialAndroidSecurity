package com.mlblab.twitterSec.androZoo

import java.io.File
import java.net.URL
import scala.sys.process._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.collection.parallel.ForkJoinTaskSupport
import java.text.DecimalFormat
import com.mlblab.twitterSec.utils.ConfigValues

object Download {
  var path = "/Users/jdeloach/Documents/ML Data/2016/"
  var downloadPath = "/Volumes/lacie0/AndroZoo/Play2016/"
  var APIKEY = ConfigValues.AndroZooAPIKEY
  var range:Range = _
  var rate = 20
  
  def main(args: Array[String]) = {
    val df = new DecimalFormat("000")
    val segment = "001"//
        
    val useSSDcacheLocal = true
    var useSSDcache = useSSDcacheLocal  
    
    if(args.length > 0) {
      println("Downloader.scala InputPath OutputPath CountUp [APIKEY] [RATE=20] [useSSDcache] [CountStartAt]")
      path = args(1)
      downloadPath = args(2)
      range = if(args(3).toBoolean) args(7).toInt until 256 else args(7).toInt to 0 by -1
      APIKEY = args(4) 
      rate = args(5).toInt
      useSSDcache = args(6).toBoolean
    }
    
    if(useSSDcacheLocal) {
      range = 188 until 200
      downloadPath = "/Users/jdeloach/data/androTmp/"
      rate = 40
    }
    
    (range).foreach(x => {downloadSegment(df.format(x)); if(useSSDcache) tarFile(df.format(x))})
  }
  
  def downloadSegment(segment: String) = {
    println("Segment: " + segment)
    val outDir = downloadPath + segment + "/"
    
    new File(outDir).mkdirs()
    
    val alreadyDownloaded = new File(outDir).listFiles.map(_.getName.replace(".apk", ""))
    
    val toCollect = scala.io.Source.fromFile(path + "toDownload.parts.txt/part-00" + segment).getLines.toList.filter(!alreadyDownloaded.contains(_)).par
    toCollect.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(rate))

    var time = System.nanoTime
    var count = 0
    var total = 0

    toCollect.foreach { sha256 => {
      if(seconds(time) > 60) {
        time = System.nanoTime
        count = 0
      }
            
      count = count + 1
      total = total + 1
      
      val url = s"https://androzoo.uni.lu/api/download?apikey=${APIKEY}&sha256=${sha256}"
      new URL(url) #> new File(outDir + sha256 + ".apk") !!
      
      if(count % 50 == 0) {
        println("running at rate: " + (count / seconds(time).toDouble) + ", total: " + total)
      }
    }}
  }
  
  def seconds(since: Long) = (System.nanoTime - since) / 1000000000.0

  def tarFile(segment: String) = {
    try {
      s"/Users/jdeloach/tarAndSSH.sh $segment".run
    }
    catch {
      case e: Exception => println("error"); e.printStackTrace()
    }
  }
}