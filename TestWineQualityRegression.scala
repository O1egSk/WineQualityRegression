import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.numerics._

object TestWineQualityRegression {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Wine Quality Regression Testing")
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
    val Array(_, testData) = output.randomSplit(Array(0.8, 0.2))

    // Конвертация данных для использования с Breeze
    val features = testData.select("features").rdd.map(row => BDV(row.getAs[org.apache.spark.ml.linalg.Vector](0).toArray)).collect()
    val labels = testData.select("label").rdd.map(row => row.getDouble(0)).collect()
    val featureMatrix = BDM(features: _*)
    val labelVector = BDV(labels)

    // Кастомный блок: стандартизация признаков
    val scaledFeatures = StandardScaler.fitTransform(featureMatrix)

    // Кастомный блок: загрузка весов модели
    val weights = BDV(spark.sparkContext.textFile("/tmp/wine_quality_regression_weights").map(_.toDouble).collect())

    // Предсказания
    val predictions = scaledFeatures * weights

    // Анализ ошибки
    val mse = mean((predictions - labelVector).map(x => pow(x, 2)))
    println(s"Mean Squared Error: $mse")

    spark.stop()
  }
}


