package com.mlblab.twitterSec

import twitter4j.Query
import twitter4j.QueryResult
import twitter4j.Twitter
import twitter4j.TwitterFactory
import twitter4j.conf.ConfigurationBuilder
import scala.collection.JavaConversions._
import twitter4j.Status
import com.mlblab.twitterSec.utils.ConfigValues

object CrawlHistoricalTweets {
  def main(args: Array[String]) : Unit = {
    val builder = new ConfigurationBuilder()
    builder.setOAuthAccessToken(ConfigValues.TwitterOAuthAccessToken)
    builder.setOAuthAccessTokenSecret(ConfigValues.TwitterOAuthAccessTokenSecret)
    builder.setOAuthConsumerKey(ConfigValues.TwitterOAuthConsumerKey)
    builder.setOAuthConsumerSecret(ConfigValues.TwitterOAuthConsumerSecret)
        
    val tf = new TwitterFactory(builder.build())
    val twitter = tf.getInstance()
    var query = new Query("url:play.google.com")
    query.setUntil("2014-12-31")
    query.setCount(100)
    println("starting query...")
    var result = twitter.search(query)
    
    println(s"result set size: ${result.getTweets.size()}")
    result.getTweets.toList.foreach(x => println(s"id: ${x.getId}, date: ${x.getCreatedAt}, text: ${x.getText}"))

    for(res <- result.getTweets) {
      println(s"id: ${res.getId}, text: ${res.getText}")
    }
    
    if(result.hasNext()) {
      result.getTweets.toList.foreach(x => println(s"id: ${x.getId}, date: ${x.getCreatedAt} text: ${x.getText}"))
      query = result.nextQuery()
      result = twitter.search(query)
      Thread.sleep(15 * 1000)
    }    
    
    println("done")
  }
}