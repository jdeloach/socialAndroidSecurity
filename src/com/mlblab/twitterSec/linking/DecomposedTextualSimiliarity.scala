package com.mlblab.twitterSec.linking

import scala.reflect.runtime.universe
import org.apache.log4j.FileAppender
import org.apache.log4j.LogManager
import org.apache.log4j.Priority
import org.apache.log4j.SimpleLayout
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf
import com.mlblab.twitterSec.utils.MathUtils
import com.mlblab.twitterSec.utils.FeatureReductionMethod._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import com.mlblab.twitterSec.utils.ClassifierUtils
import com.mlblab.twitterSec.utils.Utils._
import com.mlblab.twitterSec.utils.FeaturePrepUtils
import com.mlblab.twitterSec.utils.Utils
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import com.mlblab.twitterSec.utils.VectorizerForm._
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS

object DecomposedTextualSimiliarity {
  val path = "linkingTextEn.obj"
  val useNGram = true

  var sc: SparkContext = _
  var sql: SQLContext = _
  val log = Utils.getLogger
  
  var data: RDD[(String,(String,String))] = _
  var apks: RDD[String] = _
  
  def main(args: Array[String]) = {
    setup()
    sc.setCheckpointDir("checkpoint/")

    if(sc.isLocal)
      data = sc.parallelize(sc.objectFile[(String,(String,String))](path).take(200)).repartition(40).cache // apk -> (tweets,appstore)
    else
      data = sc.objectFile[(String,(String,String))](path).repartition(480).cache // apk -> (tweets,appstore)
    
    apks = data.map(_._1)
    val ct = apks.count.toInt
    
    val methods = Seq(PCA, SVD) // feature reduction methods
    val components = List(0/*, .25*ct, .5*ct, .75*ct*/).map(_.toInt) // number of components to reduce to with feature reduction
    val ns = Seq(/*1,2,*/3) // n's to experiment with for nGrams --3 best
    val nGramDF = Seq(2/*,3*/) // document count min for nGrams
    val NOGramDF = Seq(3,5,7/*,10,15*/) // document count min for single terms
    val w_ms = Seq(0/*, 0.01, 0.1*/) // min weight in document matrix for a term, per WeiWei WTFM paper
    val vecTypes = Seq(COUNT, BINARY, TFIDF)
    val stems = Seq(true, false)
    val ranks = List(/*20, 50,*/ 100, 200, 400, 600)
    val iterations = List(5, 10/*, 20, 30*/)
    val alphas = List(/*0.1, 0.5, 1, 2, 3,*/ 5, 20, 50, 100)
    
    ns.foreach { n => {
      val dfMins = n match {
        case 1 => NOGramDF
        case _ => nGramDF
      }
      for(dfMin <- dfMins; rank <- ranks; iteration <- iterations; alpha <- alphas) wtmf(rank, iteration, n, dfMin, alpha)    
      //for(component <- components; /*method <- methods;*/ dfMin <- dfMins; w_m <- w_ms; vecType <- vecTypes; stem <- stems) iterate(component, /*method*/SVD, n, dfMin, w_m, vecType, stem)
    }}    
  }
  
  def wtmf(rank: Int, iterations: Int, n: Int, dfMin: Int, alpha: Double, stem: Boolean = false) = {
    val docVectors = FeaturePrepUtils.createTerms(sql, data.flatMap(x => Seq(x._1 + "_t" -> x._2._1, x._1 + "_m" -> x._2._2)), n, dfMin, TFIDF, stem).zipWithIndex()
    val docIndexDb = docVectors.map { case ((key,vector),docIndex) => docIndex -> key }.collectAsMap
    val docKeyDb = docVectors.map { case ((key,vector),docIndex) => key.substring(0,key.lastIndexOf('_')) -> docIndex }.groupByKey.collectAsMap
    val boilerplateInfo = s"WTMF, iterations: $iterations, rank: $rank, alpha: $alpha, nGram: $n, docFrequencyMin:$dfMin, stemmed: $stem"

    val docRatings = docVectors.flatMap{ case ((key,vector),docIndex) => {
      val rawRatings = vector.toSparse.indices.toList.zip(vector.toSparse.values)
      rawRatings.map { case (wordIndex,value) => Rating(docIndex.toInt, wordIndex, value) }
    }}
    
    val model = ALS.trainImplicit(docRatings, rank, iterations, 0.01, alpha)
    val completedDocVectors = model.userFeatures.map{ case (idx,values) => new IndexedRow(idx,Vectors.dense(values)) }
    val matrix = new IndexedRowMatrix(completedDocVectors)
    val transposed = MathUtils.transposeIndexedRowMatrix(matrix).toRowMatrix()
    println("transposed dims: " + transposed.numRows() + "x" + transposed.numCols)
    val similarityMatrix = transposed.columnSimilarities//(.8)
    
    val ranks = similarityMatrix.toIndexedRowMatrix().rows
      .filter(row => docIndexDb(row.index.toInt).endsWith("_t"))  
      .map { twitterRow => {
        var apk = docIndexDb(twitterRow.index); apk = apk.substring(0,apk.lastIndexOf('_'))
        val otherSpot = (docKeyDb(apk).toSet - twitterRow.index).head
        val metaVal = twitterRow.vector.toArray(otherSpot.toInt)
        val metaRowRanked = twitterRow.vector.toArray.toList
                            .zipWithIndex.filter{ case (value,idx) => docIndexDb(idx).endsWith("_m") } // only rank against other metadata, don't count tweets in ranking
                            .map(_._1).sorted.reverse      
        val bestMatch = docIndexDb(twitterRow.vector.toArray.toList.indexOf(metaRowRanked.head))
                  
        LinkedResult(metaRowRanked.indexOf(metaVal), metaRowRanked.head/*metaVal*/, apk, bestMatch.substring(0, bestMatch.lastIndexOf('_')))
      }}.collect
            
      evaluateResults(ranks, boilerplateInfo)
  }
  
  def iterate(components: Int, reducer: FeatureReductionMethod, n: Int, dfMin: Int, w_m: Double, vectorizer: VectorizerForm = COUNT, stem: Boolean = false) = {
    val docVectors = FeaturePrepUtils.createTerms(sql, data.flatMap(x => Seq(x._1 + "_t" -> x._2._1, x._1 + "_m" -> x._2._2)), n, dfMin, vectorizer, stem)
    /*val docVectors = dv.mapValues { vector => Vectors.dense(vector.toDense.toArray.map { 
      case 0 => w_m
      case x => x }) }*/ //<--- this is wildly non-performant
    
    log.warn(s"terms.size: ${docVectors.first._2.size}")    
    val termsMatrix = MathUtils.transposeRowMatrix(new RowMatrix(docVectors.map(_._2)))
    
    val docVectorsIndex = docVectors.zipWithIndex.map(x => x._1._1 -> x._2.toInt).collectAsMap // apk_{t,m} -> column
    val docIndexMap = docVectors.zipWithIndex.map(x => x._2.toInt -> x._1._1).collectAsMap // column -> apk_{t,m}
    var docVectorsReduced = new IndexedRowMatrix(docVectors.map(x => new IndexedRow(docVectorsIndex(x._1),x._2)))
    val boilerplateInfo = s"reducer: $reducer, components: $components, nGram: $n, docFrequencyMin:$dfMin, W_m: $w_m, vectorizerForm: $vectorizer, stemmed: $stem"
    
    if(components != 0) {
      // create reduced space
      val reducedSpace = reducer match {
        case PCA => jointReducedSpacePCA(termsMatrix, components)
        case SVD => jointReducedSpaceSVD(termsMatrix, components)
      }
      
      val reducedSpaceLocal = new DenseMatrix(reducedSpace.rows.count.toInt, reducedSpace.rows.first.size, reducedSpace.rows.flatMap(_.toArray).collect)
      log.warn(s"dims of reducedSpace: ${reducedSpace.numRows}x${reducedSpace.numCols}, $boilerplateInfo")
      log.warn(s"dims of reducedSpaceLocal: ${reducedSpaceLocal.numRows}x${reducedSpaceLocal.numCols}, $boilerplateInfo")
      
      // transform to reduced space
      docVectorsReduced = docVectorsReduced.multiply(reducedSpaceLocal)
    }
    
    // calculate cosine similarity matrices
    val cosineSimilarityMatrix = MathUtils.transposeRowMatrix(docVectorsReduced.toRowMatrix()).columnSimilarities() // upper triangular nxn matrix
        
    val ranks = cosineSimilarityMatrix.toIndexedRowMatrix().rows
      .filter(row => docIndexMap(row.index.toInt).endsWith("_t"))
      .map { twitterRow => {
        val keyName = docIndexMap(twitterRow.index.toInt)
        val apk = keyName.substring(0, keyName.lastIndexOf('_'))
        val mCol = docVectorsIndex(apk + "_m")
        val rowRanked = twitterRow.vector.toArray.toList.sorted.reverse
        val metaRowRanked = twitterRow.vector.toArray.toList
                            .zipWithIndex.filter{ case (value,idx) => docIndexMap(idx).endsWith("_m") } // only rank against other metadata, don't count tweets in ranking
                            .map(_._1).sorted.reverse
        val metaVal = twitterRow.vector.toArray(mCol)
        val bestMatch = docIndexMap(twitterRow.vector.toArray.toList.indexOf(metaRowRanked.head))
        
        LinkedResult(metaRowRanked.indexOf(metaVal),metaVal,apk, bestMatch.substring(0, bestMatch.lastIndexOf('_')))
      }}.collect
    
    log.warn(s"dims of docVectorsReduced: ${docVectorsReduced.numRows}x${docVectorsReduced.numCols}, $boilerplateInfo")
    log.warn(s"dims of similarity matrix: ${cosineSimilarityMatrix.numRows}x${cosineSimilarityMatrix.numCols}, $boilerplateInfo")
    
    evaluateResults(ranks, boilerplateInfo)
  }
  
  def evaluateResults(ranks: Array[LinkedResult], boilerplateInfo: String) = {
    val total = apks.count
    val perfect = ranks.count(x => x.rank == 0)
    val correctAtLevels = List(.01, .03, .05, .1, .15, .2, .3).map { threshold => threshold -> ranks.filter(x => x.rank <= threshold*total).size }
        
    val (perfConf,nonPerfConf,confAvg) = (MathUtils.mean(ranks.filter(_.rank == 0).map(_.confidence)), MathUtils.mean(ranks.filter(_.rank != 0).map(_.confidence)), MathUtils.mean(ranks.map(_.confidence)))
    val confStdDev = MathUtils.stddev(ranks.map(_.confidence), confAvg)
    log.warn(s"average confidence when right: $perfConf, when wrong: $nonPerfConf, overall confidence average: $confAvg, confidence std dev: $confStdDev")
    
    val metrics = ClassifierUtils.rankedMetrics(sc.parallelize(ranks.map(x => (x.rank,x.confidence))))
    log.warn(s"auPRC: ${metrics.areaUnderPR}, $boilerplateInfo")
    //log.warn(s"precision: ${metrics.pr.collect.mkString(", ")}, $boilerplateInfo")
    
    if(metrics.areaUnderPR > .99d) {
      println(ranks.map(x => s"(${x.rank},${x.confidence})").mkString(","))
    }
    
    log.warn(s"for a perfect match, $perfect were recalled, making acc_0: ${perfect/total.toDouble}, $boilerplateInfo")
    //sc.parallelize(ranks, 1).saveAsObjectFile("results.obj")
    correctAtLevels.foreach{ case (threshold,correct) => log.warn(s"at threshold: $threshold, $correct were recalled, making acc_$threshold: ${correct/total.toDouble}, $boilerplateInfo") }
  }
  
  def jointReducedSpaceSVD(termsMatrix: RowMatrix, components: Int) : RowMatrix = {
    val svd = termsMatrix.computeSVD(components, computeU = true)
    val sMat = DenseMatrix.diag(svd.s)
    
    log.warn(s"dims of U: ${svd.U.numRows}x${svd.U.numCols}")
    log.warn(s"dims of sMat: ${sMat.numRows}x${sMat.numCols}")
    
    svd.U.multiply(sMat)
  }
  
  def jointReducedSpacePCA(termsMatrix: RowMatrix, components: Int) : RowMatrix = {
    val pca = termsMatrix.computePrincipalComponents(components)
    termsMatrix.multiply(pca)
  }
 
  def setup() = {
    var conf = if(System.getProperty("os.name").contains("OS X")) new SparkConf().setAppName(this.getClass.getSimpleName + "4gb").setMaster("local[10]") else new SparkConf().setAppName(this.getClass.getSimpleName + "4g-lite") 
    conf.set("spark.driver.maxResultSize", "4g")
    sc = new SparkContext(conf)
    sql = new SQLContext(sc)
  }
}