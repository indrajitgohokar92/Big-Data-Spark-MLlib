// Databricks notebook source
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

// COMMAND ----------

val input = sc.textFile("/FileStore/tables/h9739ehp1499868818844/Phishing_Training_Dataset-bcca3.txt")

val classMap = Map("-1" -> 0.0, "1" -> 1.0)
  val data_tmp = input.map { line =>
  val lineSplit = line.split(',')
  val values = Vectors.dense(lineSplit.take(30).map(_.toDouble))
   LabeledPoint(classMap(lineSplit(30)), values)
  }.persist()

// COMMAND ----------

val data = data_tmp.toDF("label","features")

// COMMAND ----------

data_tmp.collect().foreach(println)

// COMMAND ----------

display(data)

// COMMAND ----------

data.count()

// COMMAND ----------

val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)

// COMMAND ----------

val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
  .fit(data)

// COMMAND ----------

val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")

// COMMAND ----------

val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// COMMAND ----------

val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// COMMAND ----------

val model = pipeline.fit(trainingData)

// COMMAND ----------

val predictions = model.transform(testData)

// COMMAND ----------

predictions.select("predictedLabel", "label", "features").show(5)

// COMMAND ----------

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

// COMMAND ----------

predictions.show()

// COMMAND ----------

val predictionAndLabels = predictions.selectExpr("label","cast(predictedlabel as double) predictedlabel")
predictionAndLabels.count()

// COMMAND ----------

val predictionAndLabels1 = predictionAndLabels.rdd
.map(r => (r.getAs[Double]("label"),r.getAs[Double]("predictedlabel")))

// COMMAND ----------

import org.apache.spark.mllib.evaluation.MulticlassMetrics
val metrics = new MulticlassMetrics(predictionAndLabels1)

println("Confusion matrix:")
println(metrics.confusionMatrix)

// COMMAND ----------

val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// COMMAND ----------

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)

// COMMAND ----------

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

// COMMAND ----------

val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)


// COMMAND ----------

val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// COMMAND ----------

val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// COMMAND ----------

val model = pipeline.fit(trainingData)


// COMMAND ----------

val predictions = model.transform(testData)

// COMMAND ----------

predictions.select("predictedLabel", "label", "features").show(5)

// COMMAND ----------

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

// COMMAND ----------

val predictionAndLabels = predictions.selectExpr("label","cast(predictedlabel as double) predictedlabel")
predictionAndLabels.count()

// COMMAND ----------

val predictionAndLabels1 = predictionAndLabels.rdd
.map(r => (r.getAs[Double]("label"),r.getAs[Double]("predictedlabel")))

// COMMAND ----------

val metrics = new MulticlassMetrics(predictionAndLabels1)

println("Confusion matrix:")
println(metrics.confusionMatrix)

// COMMAND ----------

val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// COMMAND ----------

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)

// COMMAND ----------

import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors


val pca = new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(16)
  .fit(data)


// COMMAND ----------

val pcaDF = pca.transform(data)

// COMMAND ----------

display(pcaDF)

// COMMAND ----------

val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(pcaDF)

// COMMAND ----------

val featureIndexer = new VectorIndexer()
  .setInputCol("pcaFeatures")
  .setOutputCol("indexedPcaFeatures")
  .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
  .fit(pcaDF)

// COMMAND ----------

val Array(trainingData, testData) = pcaDF.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedPcaFeatures")

// COMMAND ----------

val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedPcaLabel")
  .setLabels(labelIndexer.labels)

// COMMAND ----------

val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// COMMAND ----------

val model = pipeline.fit(trainingData)

// COMMAND ----------

val predictions = model.transform(testData)

// COMMAND ----------

predictions.count()

// COMMAND ----------

predictions.select("predictedPcaLabel", "label", "pcaFeatures").show(5)

// COMMAND ----------

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

// COMMAND ----------

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

// COMMAND ----------

predictions.show()

// COMMAND ----------

val predictionAndLabels = predictions.selectExpr("label","cast(predictedPcaLabel as double) predictedPcaLabel")

// COMMAND ----------

val predictionAndLabels1 = predictionAndLabels.rdd
.map(r => (r.getAs[Double]("label"),r.getAs[Double]("predictedPcaLabel")))

// COMMAND ----------

import org.apache.spark.mllib.evaluation.MulticlassMetrics
val metrics = new MulticlassMetrics(predictionAndLabels1)

println("Confusion matrix:")
println(metrics.confusionMatrix)

// COMMAND ----------

val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// COMMAND ----------

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)

// COMMAND ----------

val Array(trainingData, testData) = pcaDF.randomSplit(Array(0.8, 0.2))

val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedPcaFeatures")
  .setNumTrees(12)

// COMMAND ----------

val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// COMMAND ----------

val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// COMMAND ----------

val model = pipeline.fit(trainingData)

// COMMAND ----------

val predictions = model.transform(testData)

// COMMAND ----------

predictions.select("predictedLabel", "label", "pcaFeatures").show(5)

// COMMAND ----------

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

// COMMAND ----------

val predictionAndLabels = predictions.selectExpr("label","cast(predictedLabel as double) predictedLabel")
predictionAndLabels.count()

val predictionAndLabels1 = predictionAndLabels.rdd
.map(r => (r.getAs[Double]("label"),r.getAs[Double]("predictedLabel")))


// COMMAND ----------

val metrics = new MulticlassMetrics(predictionAndLabels1)

println("Confusion matrix:")
println(metrics.confusionMatrix)

// COMMAND ----------

val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}


// COMMAND ----------

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)

// COMMAND ----------


