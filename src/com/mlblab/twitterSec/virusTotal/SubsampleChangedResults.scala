package com.mlblab.twitterSec.virusTotal

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object SubsampleChangedResults {
  def main(args: Array[String]) : Unit = {
    val base = "/Users/jdeloach/Developer/workspaceML/twitterMLProject/data/"
    val md5DbPath = "appIdsAndMd5sAptoide.csv"
    val oldScannerDbPath = "appIdsAndScannerCounts.csv"
    val newScannerDbPath = "partialScanResultsNov8.txt"
    
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName).setMaster("local[10]") // use 1-core locally. must set in local mode, goes to cluster choice in hadoop/yarn cluster
    val sc = new SparkContext(conf)
    // load appId->Md5, appId->ScannerCounts
    
   /*val md5Db = readInTuples(sc, base + md5DbPath)
   val oldScannerDb = readInTuples(sc, base + oldScannerDbPath)
   val newScannerDb = readInTuples(sc, base + newScannerDbPath)
   val inBothKeys = md5Db.map(_._1).toArray.toSet & oldScannerDb.map(_._1).toArray.toSet
   val combDb = md5Db.filter(x => inBothKeys.contains(x._1)).groupWith(oldScannerDb.filter(x => inBothKeys.contains(x._1))).map{ case (appId,(md5,oldScan)) => (md5.head,oldScan.head) }.collectAsMap
   
   val changeDb = newScannerDb.map{ case (md5,newScan) => (md5,combDb(md5),newScan) }
   changeDb.saveAsTextFile("changeDb.csv")*/
   val changeDb = sc.textFile("changeDb.csv").map { x => val arr = x.replace("(","").replace(")","").split(","); (arr(0),arr(1).toInt,arr(2).toInt) } 
   
   println(s"num considered: ${changeDb.count}")
   println(s"num changed: ${changeDb.filter(x => x._2 != x._3).count}")
   println(s"num downgraded: ${changeDb.filter(x => x._2 > x._3).count}")
   println(s"num upgraded: ${changeDb.filter(x => x._2 < x._3).count}")
   println(s"num neg->pos (HQ): ${changeDb.filter(x => x._2 == 0 && x._3 >= 10).count}")
   println(s"num pos->not pos (HQ): ${changeDb.filter(x => x._2 >= 10 && x._3 < 0).count}")
  }
  
  def readInTuples(sc: SparkContext, path: String) : RDD[(String,String)] = {
    sc.textFile(path).map { x => val arr = x.replace("(","").replace(")","").split(","); (arr(0),arr(1)) } 
  }  
}