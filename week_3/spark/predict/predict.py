from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.dataframe import DataFrame

from pyspark.ml import PipelineModel
from utils.constant import *


def do_prediction(df: DataFrame, info, model_name: str):
    # cargar modelo ml pyspark
    model_path = info[demo_ml_model_path] + model_name
    rf_model = PipelineModel.load(model_path)
    predictions = rf_model.transform(df)
    vector_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
    df_with_probability = predictions.withColumn('probability', vector_udf('probability')[1]) \
        .drop('features', 'rawPrediction', 'prediction')
    return df_with_probability
