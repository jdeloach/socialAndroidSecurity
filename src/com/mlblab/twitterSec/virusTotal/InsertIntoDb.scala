package com.mlblab.twitterSec.virusTotal

import com.mlblab.twitterSec.DBUtils
import org.apache.spark.sql.SQLContext
import java.sql.DriverManager
import java.io.File
import org.apache.commons.io.FilenameUtils
import scala.util.parsing.json.JSON

object InsertIntoDb {
  def main(args: Array[String]) : Unit = {
    val secResultsPath = "/Users/jdeloach/Documents/ML Data/2016/tableTwitter.txt"
    //val scanReports = "/Users/jdeloach/Developer/workspaceML/twitterMLProject/keywords/tweeeted_about_but_not_in_db2.txt/scanreports/"
    val androzooPath = "/Users/jdeloach/Documents/ML Data/2016/latest.csv"
    
    val binaryScan = readInTable(secResultsPath) // apk,features
    val scanResults =  readInAndroZooDB(androzooPath) //readInScanResults(scanReports) // apk,md5,scannersCount
    val overlap = binaryScan.keySet & scanResults.keySet
    
    val statements = overlap.map(key => s"('${scanResults(key)._3}','null',${scanResults(key)._2},'${scanResults(key)._1}',${binaryScan(key).mkString(",")})")
    //statements.take(500).foreach(println)
    insert(statements.toList)
    println("done")
  }
  
  /**
   * Reads in a table outputted from LightWeightFeatureExtractor
   */
  def readInTable(path: String) : Map[String,Array[Int]] = {
    // read file (apk.apk,features)
    scala.io.Source.fromFile(path).getLines.drop(1).map { x => {
      val splits = x.split(",")
      val apkName = FilenameUtils.getBaseName(splits(0))
      val features = splits.tail.map(_.toInt)
      (apkName,features)
    }}.toMap
  }
  
  def readInAndroZooDB(path: String) : Map[String,(String,Int,String)] = {
    scala.io.Source.fromFile(path).getLines.drop(1).map { line => {
      //sha256,sha1,md5,dex_date,apk_size,pkg_name,vercode,vt_detection,vt_scan_date,dex_size,markets
      val splits = line.split(",")
      val sha256 = splits(0)
      val md5 = splits(2)
      val pkg_name = splits(5).replace("\"","")
      val vt_detection = if(splits(7).length > 0) splits(7).toInt else -1
      
      sha256 -> (md5,vt_detection,pkg_name)
    }}.toMap
  }
  
  def readInScanResults(path: String) : Map[String,(String,Int)] = {
    val scanResults = (new File(path)).listFiles

    scanResults.flatMap(file => {
      try {
        val data = scala.io.Source.fromFile(file).getLines.mkString
        val t = JSON.parseFull(data).get.asInstanceOf[Map[String,Any]]
        Some(FilenameUtils.getBaseName(file.getName) -> (t.get("resource").get.asInstanceOf[String],t.get("positives").get.asInstanceOf[Double].toInt))
      } catch {
        case e: Exception => None
      }
    }).toMap
  }
  
  def insert(values: List[String]) = {
      Class.forName("com.mysql.jdbc.Driver");
      val connection = DriverManager.getConnection(DBUtils.mysqlURL);
      val statement = connection.createStatement
      
      // apk,class,scannersCount,md5, ... features
      
      values.grouped(200).foreach { list => {
        statement.clearBatch
        list.foreach(sql => statement.addBatch(s"insert ignore into appSec VALUES ${sql};"))        
        statement.executeBatch
      }}
      
      statement.close
      connection.close
  }
}