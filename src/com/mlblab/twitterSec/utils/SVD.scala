package com.mlblab.twitterSec.utils

import org.apache.spark.mllib.feature.VectorTransformer
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.SparseMatrix
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Matrix

/**
 * A feature transformer that projects vectors to a low-dimensional space using PCA.
 *
 * @param k number of principal components
 */
class SVD (val k: Int) {
  require(k >= 1, s"SVD requires a number of principal components k >= 1 but was given $k")

  /**
   * Computes a [[PCAModel]] that contains the principal components of the input vectors.
   *
   * @param sources source vectors
   */
  def fit(sources: RDD[Vector]): SVDModel = {
    require(k <= sources.first().size,
      s"source vector size is ${sources.first().size} must be greater than k=$k")

    val mat = new RowMatrix(sources)
    val svd = mat.computeSVD(k, computeU = true)
    new SVDModel(k, svd)
  }
}

/**
 * Model fitted by [[PCA]] that can project vectors to a low-dimensional space using PCA.
 *
 * @param k number of principal components.
 * @param pc a principal components Matrix. Each column is one principal component.
 */
class SVDModel (
    val k: Int,
    val svd: SingularValueDecomposition[RowMatrix, Matrix]) extends VectorTransformer {
  val reduced = svd.U.multiply(DenseMatrix.diag(svd.s))
  
  /**
   * Transform a vector by computed Principal Components.
   *
   * @param vector vector to be transformed.
   *               Vector must be the same length as the source vectors given to [[PCA.fit()]].
   * @return transformed vector. Vector will be of length k.
   */
  override def transform(vector: Vector): Vector = {
    reduced.multiply(Matrices.dense(1, vector.size, vector.toArray)).rows.first  
  }
}