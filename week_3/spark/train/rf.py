from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.dataframe import DataFrame

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from data_io.load import model_writer

features_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]


def do_train(df: DataFrame, info):
    df = df.withColumn('label', col('label').cast(DoubleType()).alias('label'))
    vectorAssembler = VectorAssembler().setInputCols(features_columns).setOutputCol("features")
    rf = RandomForestClassifier(numTrees=150, maxDepth=7, featureSubsetStrategy='auto')
    pipeline = Pipeline().setStages([vectorAssembler, rf])
    rf_model = pipeline.fit(df)
    model_writer(rf_model, info, "wine_rf_150.ml")
