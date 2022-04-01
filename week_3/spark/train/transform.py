from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.dataframe import DataFrame


def do_preprocess(df: DataFrame):
    df = df.withColumn("fixed acidity", col("fixed acidity").cast(DoubleType())) \
        .withColumn("volatile acidity", col("volatile acidity").cast(DoubleType())) \
        .withColumn("citric acid", col("citric acid").cast(DoubleType())) \
        .withColumn("residual sugar", col("residual sugar").cast(DoubleType())) \
        .withColumn("chlorides", col("chlorides").cast(DoubleType())) \
        .withColumn("free sulfur dioxide", col("free sulfur dioxide").cast(DoubleType())) \
        .withColumn("total sulfur dioxide", col("total sulfur dioxide").cast(DoubleType())) \
        .withColumn("density", col("density").cast(DoubleType())) \
        .withColumn("pH", col("pH").cast(DoubleType())) \
        .withColumn("sulphates", col("sulphates").cast(DoubleType())) \
        .withColumn("alcohol", col("alcohol").cast(DoubleType())) \
        .withColumn("quality", col("quality").cast(DoubleType())) \
        .withColumnRenamed('target', 'label')

    return df
