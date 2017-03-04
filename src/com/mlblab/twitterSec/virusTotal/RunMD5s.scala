package com.mlblab.twitterSec.virusTotal

import java.io.PrintWriter
import java.util.ArrayList
import scala.reflect.io.Path.string2path
import org.apache.http.NameValuePair
import org.apache.http.client.entity.UrlEncodedFormEntity
import org.apache.http.client.methods.HttpPost
import org.apache.http.impl.client.DefaultHttpClient
import org.apache.http.message.BasicNameValuePair
import com.google.gson.Gson
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import com.mlblab.twitterSec.utils.ConfigValues

object RunMD5s {
  def main(args: Array[String]) : Unit = {
    val out = new PrintWriter("scan.txt")
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
          .setMaster("local[10]") // use 1-core locally. must set in local mode, goes to cluster choice in hadoop/yarn cluster
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    
    val base = "/Users/jdeloach/Developer/workspaceML/twitterMLProject/data/"
    val toScan = readMd5List2(base + "appIdsAndMd5sAptoide.csv")
    
    val df = sqlContext.read.json(base + "scanreports/*.json")
    val toSkip = df.select(df("md5")).rdd.map(_.getString(0) -> 1).collectAsMap
    
    var time = System.nanoTime
    var count = 0
    var total = 0
    
    toScan.filter(x => !toSkip.contains(x._2)).grouped(25).toList.par.foreach{ iter => {
      if(seconds(time) < 60 && count >= 19000) {
        Thread.sleep((seconds(time) * 1000).toLong)
        time = System.nanoTime
        count = 0
      }
      val filescan = filescanReport(iter.seq.map(_._2).toArray)
      scala.tools.nsc.io.File(base + "scanreports/" + System.nanoTime + ".json").writeAll(filescan)
      count = count + 10
      total = total + 10
      
      if(count % 1000 == 0) {
        println("running at rate: " + (count / seconds(time).toDouble) + ", total: " + total)
      }
      
      if(count > 44700) {
        println("Hit 44500. Shutting down.")
        System.exit(0)
      }
    }}
  }
  
  def seconds(since: Long) = (System.nanoTime - since) / 1000000000.0
  
  def readMd5List2(path: String) : Seq[(String,String)] = {
      scala.io.Source.fromFile(path).getLines.map { x => val arr = x.split(","); (arr(0),arr(1)) }.toSeq 
  }
  
  def readMD5List(path: String) : Seq[(String,String)] = {
    scala.io.Source.fromFile(path).getLines.map { x => val arr = x.split(" "); (arr(1).substring(1, arr(1).length()-5),arr(arr.length-1)) }.toSeq
  }
  
  def filescanReport(md5s: Array[String]) : String = {
    val url = "https://www.virustotal.com/vtapi/v2/file/report"
    val client = new DefaultHttpClient
    val post = new HttpPost(url)
    val nameValuePairs = new ArrayList[NameValuePair]()
    nameValuePairs.add(new BasicNameValuePair("resource", md5s.mkString(","))); // can make this a comma seperated list
    nameValuePairs.add(new BasicNameValuePair("apikey", ConfigValues.VirusTotalApiKey));
    post.setEntity(new UrlEncodedFormEntity(nameValuePairs));
    
    // send the post request
    val response = client.execute(post)
    //response.getAllHeaders.foreach(x => println(x))
    scala.io.Source.fromInputStream(response.getEntity.getContent).getLines.mkString
  }
}