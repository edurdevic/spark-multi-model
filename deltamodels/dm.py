from pyspark.sql import functions as F
from functools import partial
from pyspark.sql.window import Window
import random
import pandas as pd
import dill
import json
import mlflow

grouped_result_cols = ["run_id", "group_key", "data", "info", "model_artifact_url", "exception"]
grouped_result_schema = "run_id string, group_key string, data string, info string, model_artifact_url string, exception string"

class GroupedModel(mlflow.pyfunc.PythonModel):
    """
    A grouped model contains a dictionary of smaller models.
    The dictionary key identifies the group key, which is used to identifi the correct model to use for inference.
    """
    def __init__(self, best_models, features):
        self.models = {}
        self.features = features
        for index, row in best_models.iterrows():
            print(f"Loading model for '{row['group_key']}' from '{row['model_artifact_url']}'")
        
            model = mlflow.pyfunc.load_model(row['model_artifact_url'])
            self.models[row['group_key']] = model

    
    def predict(self, context, dataframe: pd.DataFrame):

        dfs = []
        for group_key, df in dataframe.groupby("group_key"):
            try:
                result = pd.DataFrame(self.models[group_key].predict(df[self.features]), columns=["prediction"])
            except:
                # TODO: Handle errors better
                print(f"Error for group_key '{group_key}'")
                result = pd.DataFrame([None], columns=["prediction"])
        
            dfs.append(result)

        return pd.concat(dfs)
    
def get_best_model(df, metric: str):
    window_spec  = Window.partitionBy("group_key").orderBy(F.col(metric).desc())

    result = (df
              .filter(F.col("exception").isNull())
              .withColumn("metric", F.from_json(F.col("data"), schema=f"metrics struct<{metric}: float>"))
              .select("run_id", "group_key", "ts", "data", "model_artifact_url", f"metric.metrics.{metric}")
              .withColumn("rank", F.rank().over(window_spec))
              .filter(F.col("rank") == 1)
              .drop("rank")
    )
    return result
  
def stringify_key_value(group_key):
    """Transforms a group key into a string representation"""
    if (type(group_key) == tuple) and (len(group_key) == 1):
        return group_key[0]
    else:
        return json.dumps(group_key)
    
def train_in_parallel(df, f, parent_run, target_table):
    assert "group_key" in df.columns, "The input dataframe must contain a column 'group_key'. The models are trained independently for each distinct group key."

    res = (df
        .groupby("group_key")
        .applyInPandas(
            grouped_training(f=f, parent_run_id=parent_run.info.run_id), 
            schema=grouped_result_schema)
        .withColumn("ts", F.now()) # Adding a timestamp column to track the model creation datetime
        )
    (res.write
        .mode("append")
        .saveAsTable(target_table)
    )

def grouped_training(f, parent_run_id):
    return partial(_apply_grouped_training, f, parent_run_id)

def _apply_grouped_training(f, parent_run_id, group_key, pdf):
    import traceback
    import mlflow

    # mlflow.autolog(exclusive=True, log_models=False)
    result = None

    with mlflow.start_run(run_id=parent_run_id) as parent_run:
        with mlflow.start_run(run_name=f"group {group_key}", nested=True) as run:
            try:  
                mlflow.log_param("group_key", group_key)
                mlflow.log_param("nested_run", "true")
                mlflow.log_param("logged_model_name", "model")
                
                f(pdf, run)
            
            except Exception as e:
                trace = traceback.format_exc()
                # End run and get status
                result = pd.DataFrame(data=[[run.info.run_id, stringify_key_value(group_key), None, None, None, trace]], columns=grouped_result_cols)

    run_id = run.info.run_id
    saved_run = mlflow.get_run(run_id)
    data = json.dumps(saved_run.data.to_dictionary())
    info = json.dumps(vars(saved_run.info))

    if (result is None):
        result = pd.DataFrame(data=[[run_id, stringify_key_value(group_key), data, info, f'runs:/{run_id}/model', None]], columns=grouped_result_cols)
    
    return result
  

def grouped_prediction(model, feature_cols, id_cols):
    return partial(_apply_grouped_prediction, model, feature_cols, id_cols)

def _apply_grouped_prediction(model, feature_cols, id_cols, group_key, pdf):
    import traceback
    import mlflow

    group_key_str = stringify_key_value(group_key)
    
    result = model.predict(pdf[feature_cols])

    result_df = pdf[id_cols]
    result_df["group_key"] = group_key_str
    result_df["prediction"] = result
    
    return result_df

def get_latest_model_version(model_name):
    """Gets the latest version of a registered model"""
    mlflow_client = mlflow.MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version
