from pyspark.sql import functions as F
from functools import partial
from pyspark.sql.window import Window
import random
import pandas
import dill
import json

grouped_result_cols = ["run_id", "group_key", "data", "info", "model_artifact_url", "exception"]
grouped_result_schema = "run_id string, group_key string, data string, info string, model_artifact_url string, exception string"

def get_best_model(df, metric: str):
    window_spec  = Window.partitionBy("group_key").orderBy(F.col(metric).desc())

    result = (df
              .filter(F.col("exception").isNull())
              .withColumn("metric", F.from_json(F.col("data"), schema=f"metrics struct<{metric}: float>"))
              .select("run_id", "group_key", "ts", "workflow_run_id", "data", "model_artifact_url", f"metric.metrics.{metric}")
              .withColumn("rank", F.rank().over(window_spec))
              .filter(F.col("rank") == 1)
              .drop("rank")
    )
    return result
  
def grouped_training(f, model_name):
    return partial(apply_grouped_training, f, model_name)

def apply_grouped_training(f, model_name, group_key, pdf):
    import traceback
    import mlflow

    # mlflow.autolog(exclusive=True, log_models=False)
    result = None

    with mlflow.start_run() as run:
        try:  
            f(pdf, run)
        
        except Exception as e:
            trace = traceback.format_exc()
            # End run and get status
            result = pandas.DataFrame(data=[[run.info.run_id, json.dumps(group_key), None, None, None, trace]], columns=grouped_result_cols)

    run_id = run.info.run_id
    saved_run = mlflow.get_run(run_id)
    data = json.dumps(saved_run.data.to_dictionary())
    info = json.dumps(vars(saved_run.info))

    if (result is None):
        result = pandas.DataFrame(data=[[run_id, json.dumps(group_key), data, info, f'runs:/{run_id}/{model_name}', None]], columns=grouped_result_cols)
    
    return result
  
def grouped_prediction(best_models, feature_cols, id_cols):
    return partial(apply_grouped_prediction, best_models, feature_cols, id_cols)

def apply_grouped_prediction(best_models, feature_cols, id_cols, group_key, pdf):
    import traceback
    import mlflow

    group_key_str = json.dumps(group_key)
    model_url = best_models[best_models["group_key"] == group_key_str]["model_artifact_url"].iloc[0]
    model = mlflow.pyfunc.load_model(model_url)

    result = model.predict(pdf[feature_cols])

    result_df = pdf[id_cols]
    result_df["group_key"] = group_key_str
    result_df["model_url"] = model_url
    result_df["prediction"] = result
    
    return result_df
  

