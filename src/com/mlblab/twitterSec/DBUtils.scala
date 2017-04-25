package com.mlblab.twitterSec

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import twitter4j.Status
import twitter4j.TwitterObjectFactory
import org.apache.commons.io.FilenameUtils
import scala.reflect.runtime.Settings
import com.mlblab.twitterSec.utils.ConfigValues
import com.mlblab.twitterSec.utils.Utils
import com.mlblab.twitterSec.utils.Utils._
import com.mlblab.twitterSec.utils.MislabeledLabeledPoint

object DBUtils {
  val mysqlURL = ConfigValues.MysqlURL
  
  def appExistsInDb(sc: SparkContext, appID: String) : Boolean = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val jdbcDF = sqlContext.read.format("jdbc").options(
      Map("url" -> mysqlURL,
      "dbtable" -> "APPS2")).load()
      
    jdbcDF.filter(s"APP_ID = '$appID'").count > 0
  }
  
  def appsExistInDb(sc: SparkContext, appIds: List[String]) : List[Boolean] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val jdbcDF = sqlContext.read.format("jdbc").options(
      Map("url" -> mysqlURL,
      "dbtable" -> "APPS2")).load()
    
    appIds.map { appID => jdbcDF.filter(s"APP_ID = '$appID'").count > 0 }
  }
  
  def loadTwitterFile(sc: SparkContext, path: String) : RDD[Status] = {
    sc.textFile(path).map { x => TwitterObjectFactory.createStatus(x) }
  }
  
  def loadPlaystoreUrls(sqlContext: SQLContext, path: String) : RDD[String] = {
    sqlContext.read.json(path).select("urlEntities.expandedURL").flatMap(_.getSeq[String](0))
  }
  
  def getTweetAndApiIdPairs(sqlContext: SQLContext, path: String) : (RDD[(String,String)],Long) = {
    val df = sqlContext.read.json(path)
    df.registerTempTable("tweets")
    (sqlContext.sql("select text,urlEntities.expandedURL from tweets").map(x => (x.getString(0),x.getSeq[String](1))).flatMap{ case (text,urls) => urls.filter(_.contains("play.google")).map(url => (text,appIdFromUrl(url))) },df.count)
  }
  
  /**
   * Uses an expanded search to find 
   */
  def loadSelectData(sc: SparkContext) : RDD[(String,LabeledPoint)] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    import org.apache.spark.sql.functions.lit

    val appSecDF = sqlContext.read.format("jdbc").options(
      Map("url" -> mysqlURL,
      "dbtable" -> "appSecWithAppIdExtended")).load()
        
    appSecDF./*where(appSecDF("APP_ID").isin(appIds:_*)).*/filter("scannersCount != -1").map { row => {
      val label = row.getInt(row.fieldIndex("scannersCount")) match {
        case x if x >= 3 => 1d
        case x if x == 0 => 0d
        case _ => -1d
      }
      
      val features = Vectors.dense(row.toSeq.drop(3).map { _.asInstanceOf[Integer].toDouble }.toArray)
      val appID = row.getString(row.fieldIndex("APP_ID"))
      ((appID),new LabeledPoint(label,features))
    }}.filter { x => x._2.label != -1 }      
  }
  
  def loadSelectData2(sc: SparkContext, appIds: Seq[String]) : RDD[(String,LabeledPoint)] = {
    val md5DB = sc.textFile("data/appIdsAndMd5sAptoide.csv") // appIdsAndMd5sAndroZooTwitter.csv
                      .map{x => val a = x.split(","); (a(0),a(1))}
                      .collectAsMap
    val appIdDB = md5DB.map(x => x._2 -> x._1).toMap
    val md5s = appIds.flatMap(x => if (md5DB.contains(x)) Some(md5DB(x)) else None)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    import org.apache.spark.sql.functions.lit

    val appSecDF = sqlContext.read.format("jdbc").options(
      Map("url" -> mysqlURL,
      "dbtable" -> "appSec")).load()
      
    val t = appSecDF.where(appSecDF("md5").isin(md5s:_*)).filter("scannersCount != -1").map { row => {
      val label = row.getInt(row.fieldIndex("scannersCount")) match {
        case x if x >= 5 => 1d
        case x if x == 0 => 0d
        case _ => -1d
      }
      
      val features = Vectors.dense(row.toSeq.drop(4).map { _.asInstanceOf[Integer].toDouble }.toArray)
      val appID = appIdDB(row.getString(row.fieldIndex("md5")))
      ((appID),new LabeledPoint(label,features))
    }}.cache
                      
    Utils.getLogger.warn(s"appIds.count: ${appIds.size}, md5s.count: ${md5s.size}, pos: ${t.filter(_._2.label == 1).count}, neg: ${t.filter(_._2.label == 0).count}, invalid: ${t.filter(_._2.label == -1).count}")
    val missingAppIds = appIds.toSet -- t.keys.collect.toSet
    Utils.getLogger.warn(s"missingAppIds.size: ${missingAppIds.size}, list: ${missingAppIds.mkString(",")}")
    t.filter { x => x._2.label != -1 }  
  }
  
  def load2016LabelsSimple(sc: SparkContext) : RDD[(String,LabeledPoint)] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val jdbcDF = sqlContext.read.format("jdbc").options(
      Map("url" -> mysqlURL,
      "dbtable" -> "appSec")).load()    
    
    jdbcDF.filter("scannersCount != -1").map { row => {
      val label = row.getInt(row.fieldIndex("scannersCount")) match {
        case x if x >= 5 => 1d
        case x if x == 0 => 0d
        case _ => -1d
      }
      
      val features = Vectors.dense(row.toSeq.drop(4).map { _.asInstanceOf[Integer].toDouble }.toArray)
      val apkName = row.getString(row.fieldIndex("apk"))
      val appId = if (apkName.contains("-"))  apkName.substring(0, apkName.indexOf('-')) else ""
      (appId,new LabeledPoint(label,features))
    }}.filter { x => x._2.label != -1 }          
  }
  
  def readInAndroZooFeatureResults(resultsPath: String, sc: SparkContext) : RDD[LabeledPoint] = {
    def readInAndroZooDB(path: String) : Map[String,Int] = {
        scala.io.Source.fromFile(path).getLines.drop(1).map { line => {
        //sha256,sha1,md5,dex_date,apk_size,pkg_name,vercode,vt_detection,vt_scan_date,dex_size,markets
        val splits = line.split(",")
        val sha256 = splits(0)
        val md5 = splits(2)
        val pkg_name = splits(5).replace("\"","")
        val vt_detection = if(splits(7).length > 0) splits(7).toInt else -1
        
        sha256 -> vt_detection
      }}.toMap
    }
    
    val androzooDb = readInAndroZooDB("/Users/jdeloach/Documents/ML Data/2016/latest.csv")
    
    readInTable(resultsPath, sc)
      .map{ case (sha256,features) => {
        val label = androzooDb(sha256) match {
          case x if x >= 10 => 1d
          case x if x == 0 => 0d
          case _ => -1d
        }
        new LabeledPoint(label,Vectors.dense(features))
      }}.filter(_.label != -1)
  }
  
  def readInTable(path: String, sc: SparkContext) : RDD[(String,Array[Double])] = {
    // read file (apk.apk,features)
    sc.textFile(path).flatMap { x => {
      if(x.startsWith("apk") || x.endsWith("ACTION_PACKAGE_RESTARTED")) {
        None
      } else {
        val splits = x.split(",")
        val apkName = FilenameUtils.getBaseName(splits(0))
        val features = splits.tail.map(_.toDouble)
        Some(apkName,features)
      }
    }}
  }
  
  def appIdFromApkName(x: String) = if (x.contains("-"))  x.substring(0, x.indexOf('-')) else ""
  def appIdFromUrl(x: String) = (if (x.indexOf('&', x.indexOf('=')) > 0) x.substring(x.indexOf('=')+1, x.indexOf('&', x.indexOf('='))) else x.substring(x.indexOf('=')+1))
}