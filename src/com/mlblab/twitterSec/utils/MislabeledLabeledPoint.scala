package com.mlblab.twitterSec.utils

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector

class MislabeledLabeledPoint(val realLabel: Double, misLabel: Double, features: Vector) extends LabeledPoint(misLabel, features) {
  def this(labeledPoint: LabeledPoint, realLabel: Double) { this(realLabel, labeledPoint.label, labeledPoint.features) }
}