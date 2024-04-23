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
import mlflow

# COMMAND ----------

# DBTITLE 1,Date Range Widget Getter

class Conf:
    catalog = "temp"
    schema = "erni"
    model_table = "delta_models_v1"
    grouped_model_name = "windfarm_grouped_model"
    feature_table = "windfarm_features_v1"
    registered_model_name=f"{catalog}.{schema}.{grouped_model_name}"

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
    # Define the single model training code here
    # This will be executed for each group_key

    n_estimators=5
    mlflow.log_param("n_estimators", n_estimators)
    
    regressor = LGBMRegressor(n_estimators=n_estimators, n_jobs=1)
    model = regressor.fit(pdf[["a", "b", "c"]], pdf["target"])
    
    mlflow.log_metric("rmse", random.random())
    mlflow.lightgbm.log_model(model, "model")
    
    #### END YOUR CODE #####



with mlflow.start_run() as parent_run:
    
    #### YOUR CODE #####
    # Log anything you wan to the MLflow parent run

    training_df = training_set.load_df() 
    target_table = f"{conf.catalog}.{conf.schema}.{conf.model_table}"

    mlflow.log_param("target_table", target_table)

    #### END YOUR CODE #####

    dm.train_in_parallel(
        df=training_df, # This dataset must contain a "group_key" column
        f=my_fun, # This function will be executed for each group_key
        parent_run=parent_run,
        target_table=target_table
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore results

# COMMAND ----------

# Load directly from MLflow experiment
mlflow_runs_df = spark.read.format("mlflow-experiment").load()
mlflow_runs_df.display()

# COMMAND ----------

mlflow_runs_df.groupBy("params.group_key").count().display()

# COMMAND ----------

# DBTITLE 1,Spark SQL Display Query
# Read results from the delta table
spark.sql(f"SELECT * FROM {conf.catalog}.{conf.schema}.{conf.model_table}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Count of errors per run

# COMMAND ----------

# DBTITLE 1,Workflow Exception Model Count Query
spark.sql(f"""
          SELECT ts, count(exception), count(model_artifact_url)
          FROM {conf.catalog}.{conf.schema}.{conf.model_table}
          GROUP BY 1
          ORDER BY 1
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

from deltamodels.dm import GroupedModel

mlflow.set_registry_uri('databricks-uc')

with mlflow.start_run() as run:

  # Package multiple models into a single GroupedModel artifact
  model = GroupedModel(best_models.toPandas(), features=["a", "b", "c"])

  fe.log_model(
    model=model,
    artifact_path=conf.grouped_model_name,
    flavor=mlflow.pyfunc,
    training_set=training_set,
    registered_model_name=f"{conf.registered_model_name}"
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------


fe = FeatureEngineeringClient()

# TODO: get the latest version or use an alias (eg. `prod`)
model_version = dm.get_latest_model_version(conf.registered_model_name)
print(f"Loading model version {model_version}")

predictions = fe.score_batch(
    model_uri=f"models:/{conf.registered_model_name}/{model_version}",
    df=df.select("farm_code", "ent_code", "ts")
)

# COMMAND ----------

predictions.display()

# COMMAND ----------


