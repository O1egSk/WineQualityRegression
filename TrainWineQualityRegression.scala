import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

// Кастомный блок: вычисление градиента
object Gradient {
  def computeGradient(
      features: BDM[Double],
      labels: BDV[Double],
      weights: BDV[Double]): BDV[Double] = {
    val numExamples = features.rows
    val predictions = features * weights
    val residuals = predictions - labels
    val gradient = (features.t * residuals) / numExamples.toDouble
    gradient
  }
}

// Кастомный блок: оптимизация (градиентный спуск)
object Optimizer {
  def gradientDescent(
      features: BDM[Double],
      labels: BDV[Double],
      maxIter: Int): BDV[Double] = {
    val numFeatures = features.cols
    var weights = BDV.zeros[Double](numFeatures)

    for (i <- 0 until maxIter) {
      val gradient = Gradient.computeGradient(features, labels, weights)
      weights -= gradient
    }
    weights
  }
}

// Кастомный блок: стандартизация данных
object StandardScaler {
  def fitTransform(features: BDM[Double]): BDM[Double] = {
    val mean = BDV.zeros[Double](features.cols)
    val stdDev = BDV.ones[Double](features.cols)

    for (i <- 0 until features.cols) {
      mean(i) = features(::, i).sum / features.rows
      stdDev(i) = math.sqrt(features(::, i).map(x => math.pow(x - mean(i), 2)).sum / features.rows)
    }

    val scaledFeatures = (features(*, ::) - mean) /:/ stdDev
    scaledFeatures
  }
}

// Кастомный блок: обновление весов
object Updater {
  def updateWeights(
      weights: BDV[Double],
      gradient: BDV[Double],
      learningRate: Double): BDV[Double] = {
    weights - (gradient * learningRate)
  }
}

object TrainWineQualityRegression {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Wine Quality Regression Training")
      .master("local[*]")
      .getOrCreate()

    // Загрузка данных
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("sep", ";")
      .csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")

    // Подготовка данных
    val featureColumns = Array("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol")
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")
    val output = assembler.transform(data)
      .select(col("quality").cast("double").as("label"), col("features"))

    // Разделение данных
    val Array(trainingData, _) = output.randomSplit(Array(0.8, 0.2))

    // Конвертация данных для использования с Breeze
    val features = trainingData.select("features").rdd.map(row => BDV(row.getAs[org.apache.spark.ml.linalg.Vector](0).toArray)).collect()
    val labels = trainingData.select("label").rdd.map(row => row.getDouble(0)).collect()
    val featureMatrix = BDM(features: _*)
    val labelVector = BDV(labels)

    // Кастомный блок: стандартизация признаков
    val scaledFeatures = StandardScaler.fitTransform(featureMatrix)

    // Кастомный блок: обучение модели с использованием кастомного градиентного спуска
    val maxIter = 10
    val weights = Optimizer.gradientDescent(scaledFeatures, labelVector, maxIter)

    // Сохранение обученной модели
    spark.sparkContext.parallelize(weights.toArray).saveAsTextFile("/tmp/wine_quality_regression_weights")

    spark.stop()
  }
}
