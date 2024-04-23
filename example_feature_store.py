# Databricks notebook source
# MAGIC %md
# MAGIC # Multi model training and inference
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# DBTITLE 1,Python Data Processing Libraries
from pyspark.sql import functions as F
from deltamodels import dm
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import random
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Date Range Widget Getter

class Conf:
    catalog = "temp"
    schema = "erni"
    model_table = "delta_models_v4"
    feature_table = "windfarm_features_v2"

conf = Conf()

# COMMAND ----------

# DBTITLE 1,Adaptive Query Configuration Disable
# Disabling AQE to always have 200 concurrent tasks. 
# AQE would coalesce to 1 task for small dataframes
spark.conf.set("spark.sql.adaptive.enabled", "false")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example dataset
# MAGIC

# COMMAND ----------

# DBTITLE 1,Randomized Entity Farm Data Generator
df = (spark.range(1000)
    .withColumn("ent_code", F.concat(F.lit("ent_"), (F.rand(seed=5)*3).cast("int").cast("string")))
    .withColumn("farm_code", F.concat(F.lit("farm_"), (F.rand(seed=1000)*3).cast("int").cast("string")))

    # NOTE: there must be a column called "group_key" to distinguish training groups
    .withColumn("group_key", F.concat(F.col("ent_code"), F.lit("_"), F.col("farm_code")))
    .withColumn("ts", F.col("id").cast("timestamp"))
    .withColumn("a", F.rand(seed=1)*10)
    .withColumn("b", F.rand(seed=2)*10)
    .withColumn("c", F.rand(seed=3)*10)
    .withColumn("target", F.rand(seed=2)*100)
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature engineering
# MAGIC

# COMMAND ----------

fe = FeatureEngineeringClient()

# Creating the feature engineering table
feature_table = fe.create_table(
  name=f'{conf.catalog}.{conf.schema}.{conf.feature_table}',
  primary_keys=["ent_code", "farm_code", "ts"],
  timeseries_columns='ts',
  df=df.drop("target"),
  description='Wind turbine features'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature lookup and training

# COMMAND ----------

feature_lookups = [
    FeatureLookup(
      table_name=f'{conf.catalog}.{conf.schema}.{conf.feature_table}',
      # 'group_key' is a feature because it is used to switch between models
      feature_names=['group_key', 'a', 'b', 'c'],
      lookup_key=["ent_code", "farm_code"],
      timestamp_lookup_key=["ts"]
    )
  ]

fe = FeatureEngineeringClient()

training_set = fe.create_training_set(
        df=df.select("ent_code", "farm_code", "ts", "target"),
        feature_lookups = feature_lookups,
        label = 'target',
        exclude_columns = ['id']
    )

# COMMAND ----------

training_set.load_df().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training

# COMMAND ----------

# DBTITLE 1,Grouped Regressor Training Logger
from lightgbm import LGBMRegressor
import mlflow


def my_fun(pdf: pd.DataFrame, run: mlflow.ActiveRun):

    #### YOUR CODE #####

    n_estimators=5
    mlflow.log_param("n_estimators", n_estimators)
    
    regressor = LGBMRegressor(n_estimators=n_estimators, n_jobs=1)
    model = regressor.fit(pdf[["a", "b", "c"]], pdf["target"])
    
    mlflow.log_metric("rmse", random.random())
    mlflow.lightgbm.log_model(model, "model")
    
    #### END YOUR CODE #####


with mlflow.start_run() as parent_run:
    res = (training_set.load_df()
        .groupby("group_key")
        .applyInPandas(
            dm.grouped_training(f=my_fun, parent_run_id=parent_run.info.run_id), 
            schema=dm.grouped_result_schema)
        .withColumn("ts", F.now()) # Adding a timestamp column to track the model creation datetime
        )
    (res.write
        .mode("append")
        .saveAsTable(f"{conf.catalog}.{conf.schema}.{conf.model_table}")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore results

# COMMAND ----------

# DBTITLE 1,Spark SQL Display Query
spark.sql(f"SELECT * FROM {conf.catalog}.{conf.schema}.{conf.model_table}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Count of errors per run

# COMMAND ----------

# DBTITLE 1,Workflow Exception Model Count Query
spark.sql(f"""
          SELECT ts, count(exception), count(model_artifact_url)
          FROM {conf.catalog}.{conf.schema}.{conf.model_table}
          GROUP BY 1, 2
          ORDER BY 1, 2
          """).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Details of a specific group

# COMMAND ----------

# DBTITLE 1,Show all models for a specific group
spark.sql(f"""
          SELECT *
          FROM {conf.catalog}.{conf.schema}.{conf.model_table}
          WHERE group_key = 'ent_1_farm_0'
          ORDER BY ts DESC
          """).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find best model

# COMMAND ----------

# DBTITLE 1,Metric Data Stream Filter

all_models = spark.read.table(f"{conf.catalog}.{conf.schema}.{conf.model_table}")
best_models = dm.get_best_model(all_models, metric="rmse")

best_models.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Package model
# MAGIC

# COMMAND ----------

import mlflow
import dill


class GroupedModel(mlflow.pyfunc.PythonModel):

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




# COMMAND ----------

# MAGIC %md
# MAGIC ## Log model

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')

with mlflow.start_run() as run:

  # Package multiple models into a single GroupedModel artifact
  model = GroupedModel(best_models.toPandas(), features=["a", "b", "c"])

  fe.log_model(
    model=model,
    artifact_path="deltamodels_grouped_model",
    flavor=mlflow.pyfunc,
    training_set=training_set,
    registered_model_name="temp.erni.deltamodels_grouped_model"
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------


fe = FeatureEngineeringClient()

predictions = fe.score_batch(
    model_uri="models:/temp.erni.deltamodels_grouped_model/16",
    df=df.select("farm_code", "ent_code", "ts")
)

# COMMAND ----------

predictions.display()

# COMMAND ----------


