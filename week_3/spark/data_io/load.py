from utils.constant import *

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame


def get_data(spark: SparkSession, path: str, is_csv=False) -> DataFrame:
    if not is_csv:
        df = spark.read.parquet(path)
    else:
        df = spark.read.csv(path, header=True)
    return df


def df_writer(df: DataFrame, path: str, to_csv_coalesce=False):
    if not to_csv_coalesce:
        df.write.mode('overwrite').parquet(path)
    else:
        df.coalesce(1).write.mode('overwrite').csv(path, header=True)


def df_shape(df: DataFrame):
    print((df.count(), len(df.columns)))
    print("SHAPE OF DATAFRAME: (ABOVE)")


def model_writer(rf_model, info, model_name):
    model_path = info[demo_ml_model_path] + model_name
    rf_model.save(model_path)
