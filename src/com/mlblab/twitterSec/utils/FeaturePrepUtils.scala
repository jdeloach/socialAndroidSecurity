package com.mlblab.twitterSec.utils

import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import com.mlblab.twitterSec.utils.VectorizerForm._
import org.apache.spark.ml.feature.Binarizer2
import org.apache.spark.ml.feature.PorterStemmer

object FeaturePrepUtils {
/**
   * @param sqlContext 
   * @param texts the (appID -> text) mappings
   * @param n the N to use in N-Gram. Can be 1, which will just skip n-grams
   * @param dfMin document frequency min to use in the CountVectorizer
   * @param vectorizer either 'tfidf' or 'binarizer' or something else (CountVectorizer) to determine how features are constructed
   * @param stem - whether to use a Porter stemmer or not
   */
  def createTerms(sql: SQLContext, texts: RDD[(String,String)], n:Int, dfMin: Int, vectorizer: VectorizerForm = COUNT, stem: Boolean = false) : RDD[(String,Vector)] = {
    val df = sql.createDataFrame(texts).toDF("appID", "tweetText")
    val linkedTweetsSeperated = new Tokenizer().setInputCol("tweetText").setOutputCol("words").transform(df)
    val linkedTweetsCleaned = (new StopWordsRemover()).setInputCol("words").setOutputCol("filtered").transform(linkedTweetsSeperated)
    import org.apache.spark.sql.functions._
    
    // replacement functions
    val dropLinks = udf[Seq[String],Seq[String]] (_.map(x => if(x.startsWith("http")) "<url>" else x))
    val replaceUserNames = udf[Seq[String],Seq[String]] (_.map(x => if(x.startsWith("@")) "<username>" else x))
    val dropNumbers = udf[Seq[String],Seq[String]] (_.filter(!_.forall(_.isDigit)))
    val removeHashtagSymbol = udf[Seq[String],Seq[String]] (_.map(_.replaceAll("#", "")))
    val removeNonAscii = udf[Seq[String],Seq[String]] (_.map(_.replaceAll("[^\\x00-\\x7F]", "")))
    val removeNonEnglish = udf[Seq[String],Seq[String]] (_.map(_.replaceAll("[^\\p{L}\\p{Nd}]+", "")))

    val linkedTweetsCleanedHtttp = linkedTweetsCleaned.withColumn("filteredMod", 
        removeNonEnglish(
        removeNonAscii(
        removeHashtagSymbol(
        dropNumbers(
        replaceUserNames(
        dropLinks(col("filtered"))))))))

    var currOutputField = "filteredMod"
    var data = linkedTweetsCleanedHtttp
    
    if(stem) {
      val stemmer = new PorterStemmer().setInputCol(currOutputField).setOutputCol("stemmed")
      data = stemmer.transform(data)
      currOutputField = "stemmed"
    }
    
    // n-gram or not
    data = if (n > 1) {
      val ngram = new NGram().setN(n).setInputCol(currOutputField).setOutputCol("filteredModNGram")
      // Quick Fix for local install of 1.6.0, due to SPARK-12746, make output column nullable
      val ngramTransformed = MathUtils.setNullableStateOfArrayColumn(ngram.transform(data), "filteredModNGram", true)
      val cvModel = new CountVectorizer()/*.setMinDF(dfMin)*/.setVocabSize(10000).setInputCol("filteredModNGram").setOutputCol("features").fit(ngramTransformed)
      currOutputField = "features"
      cvModel.transform(ngramTransformed)
    } else {      
      val cvModel = new CountVectorizer().setMinDF(dfMin).setInputCol(currOutputField).setOutputCol("features").fit(data)
      currOutputField = "features"
      cvModel.transform(data)
    }
    
    // a count vector is the input to the following optional transformations
    if(vectorizer == TFIDF) {
      val idf = new IDF().setInputCol("features").setOutputCol("features_final")
      val idfModel = idf.fit(data)
      data = idfModel.transform(data)
      currOutputField = "features_final"
    } else if (vectorizer == BINARY) {
      val binarizer = new Binarizer2().setInputCol("features").setOutputCol("features_final").setThreshold(0.99)
      data = binarizer.transform(data)
      currOutputField = "features_final"
    }
    
    data.select("appID", currOutputField).rdd.map(x => x.getString(0) -> x.getAs[Vector](currOutputField))
  }
  
  def cleanStrings(sql: SQLContext, texts: RDD[(String, String)]) : RDD[(String,Seq[String])] = {
    val df = sql.createDataFrame(texts).toDF("appID", "tweetText")
    val linkedTweetsSeperated = new Tokenizer().setInputCol("tweetText").setOutputCol("words").transform(df)
    val linkedTweetsCleaned = (new StopWordsRemover()).setInputCol("words").setOutputCol("filtered").transform(linkedTweetsSeperated)
    import org.apache.spark.sql.functions._
    
    // replacement functions
    val dropLinks = udf[Seq[String],Seq[String]] (_.map(x => if(x.startsWith("http")) "<url>" else x))
    val replaceUserNames = udf[Seq[String],Seq[String]] (_.map(x => if(x.startsWith("@")) "<username>" else x))
    val dropNumbers = udf[Seq[String],Seq[String]] (_.filter(!_.forall(_.isDigit)))
    val removeHashtagSymbol = udf[Seq[String],Seq[String]] (_.map(_.replaceAll("#", "")))
    val removeNonAscii = udf[Seq[String],Seq[String]] (_.map(_.replaceAll("[^\\x00-\\x7F]", "")))
    val removeNonEnglish = udf[Seq[String],Seq[String]] (_.map(_.replaceAll("[^\\p{L}\\p{Nd}]+", "")))

    val linkedTweetsCleanedHtttp = linkedTweetsCleaned.withColumn("filteredMod", 
        removeNonEnglish(
        removeNonAscii(
        removeHashtagSymbol(
        dropNumbers(
        replaceUserNames(
        dropLinks(col("filtered"))))))))

    linkedTweetsCleanedHtttp.select("appID", "filteredMod").map(x => x.getString(0) -> x.getSeq[String](1))
  }
  
  def convertToInts(texts: RDD[(String,Seq[String])]) : RDD[(String,Seq[(Int,Int)])] = {
    val wordIndex = texts.flatMap(_._2.distinct).distinct.zipWithIndex.collectAsMap
    
    texts.mapValues { words => {
      words.zipWithIndex.map { case (word,idx) => (idx,wordIndex(word).toInt) }  
    }}
  }
}