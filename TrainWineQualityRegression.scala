import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

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

    // Стандартизация признаков
    val scaledFeatures = StandardScaler.fitTransform(featureMatrix)

    // Обучение модели с использованием кастомного градиентного спуска
    val maxIter = 10
    val weights = Optimizer.gradientDescent(scaledFeatures, labelVector, maxIter)

    // Сохранение обученной модели
    // В реальном проекте вы бы сериализовали weights и другие параметры модели на диск
    spark.sparkContext.parallelize(weights.toArray).saveAsTextFile("/tmp/wine_quality_regression_weights")

    spark.stop()
  }
}
