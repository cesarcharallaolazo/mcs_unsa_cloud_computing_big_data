from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

from utils.constant import *
from data_io import load
from train import rf
from train import transform
from predict import predict
import argparse


def parse_cli_args():
    """
    Parse cli arguments
    returns a dictionary of arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', action='store', dest=root_path, type=str,
                        help='Store', default=None)

    parser.add_argument('--checkpoint_path', action='store', dest=checkpoint_path, type=str,
                        help='Store', default=None)

    parser.add_argument('--app_env', action='store', dest=app_env, type=str,
                        help='Store', default=None)

    parser.add_argument('--version', action='store', dest=version, type=str,
                        help='Store', default=None)

    parser.add_argument('--org', action='store', dest=org, type=str,
                        help='Store', default=None)

    known_args, unknown_args = parser.parse_known_args()
    known_args_dict = vars(known_args)
    return known_args_dict


if __name__ == '__main__':
    args = parse_cli_args()

    # demo
    args[demo_raw_path] = f"{args[root_path]}/raw/"
    args[demo_raw_csv_path] = f"{args[root_path]}/raw_csv/"
    args[demo_ml_model_path] = f"{args[root_path]}/ml_model/"
    args[demo_predictions_csv_path] = f"{args[root_path]}/predictions_csv/"

    # Start Spark Environment
    spark = SparkSession.builder.getOrCreate()

    # Checkpointing tuning strategy
    sc = SparkContext.getOrCreate()
    sc.setCheckpointDir(args[checkpoint_path])

    #### Pyspark ML Demo

    # **** ENTRENAMIENTO ****

    # leer csv de datos
    df = load.get_data(spark, args[demo_raw_csv_path] + "wine-quality.csv", is_csv=True)

    # procesar datos
    df = transform.do_preprocess(df)

    # crear modelo con libreria pyspark ml
    rf.do_train(df, args)

    # **** PREDICCION ****
    df_predictions = predict.do_prediction(df.drop("label"), args, "wine_rf_150.ml")

    # guardar dataset de predicciones
    load.df_writer(df_predictions, args[demo_predictions_csv_path] + "pred_rf_wine_rf_150/wine_predictions.csv",
                   to_csv_coalesce=True)
