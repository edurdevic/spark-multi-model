# Databricks notebook source
# dbutils.widgets.removeAll()
dbutils.widgets.text("workflow_run_id", "1")

# COMMAND ----------

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
import random
import pandas

# COMMAND ----------

# DBTITLE 1,Date Range Widget Getter
workflow_run_id = dbutils.widgets.get("workflow_run_id")

class Conf:
    catalog = "temp"
    schema = "erni"
    model_table = "delta_models"
    model_name = "logged_model_name"

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
    .withColumn("a", F.rand(seed=1)*10)
    .withColumn("b", F.rand(seed=2)*10)
    .withColumn("c", F.rand(seed=3)*10)
    .withColumn("target", F.rand(seed=2)*100)
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training

# COMMAND ----------

# DBTITLE 1,Grouped Regressor Training Logger
from lightgbm import LGBMRegressor
import mlflow

def my_fun(pdf: pandas.DataFrame, run: mlflow.ActiveRun):

    #### YOUR CODE #####

    n_estimators=5
    mlflow.log_param("n_estimators", n_estimators)
    
    regressor = LGBMRegressor(n_estimators=n_estimators, n_jobs=1)
    model = regressor.fit(pdf[["a", "b", "c"]], pdf["target"])
    
    mlflow.log_metric("rmse", random.random())
    mlflow.lightgbm.log_model(model, conf.model_name)
    
    #### END YOUR CODE #####


res = (df
       .groupby("farm_code", "ent_code")
       .applyInPandas(dm.grouped_training(f=my_fun, model_name=conf.model_name), schema=dm.grouped_result_schema)
       .withColumn("ts", F.now())
       .withColumn("workflow_run_id", F.lit(workflow_run_id))
    ) 
                                           
(res.write
       .mode("append")
       .saveAsTable(f"{conf.catalog}.{conf.schema}.{conf.model_table}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore results

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM ${var.catalog}

# COMMAND ----------

# DBTITLE 1,Spark SQL Display Query
spark.sql(f"SELECT * FROM {conf.catalog}.{conf.schema}.{conf.model_table}").display()

# COMMAND ----------

# DBTITLE 1,Workflow Exception Model Count Query
spark.sql(f"""
          SELECT ts, workflow_run_id, count(exception), count(model_artifact_url)
          FROM {conf.catalog}.{conf.schema}.{conf.model_table}
          GROUP BY 1, 2
          ORDER BY 1, 2
          """).display()

# COMMAND ----------

# DBTITLE 1,Show all models for a specific group
spark.sql(f"""
          SELECT *
          FROM {conf.catalog}.{conf.schema}.{conf.model_table}
          WHERE group_key = '["farm_1", "ent_1"]'
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
# MAGIC ## Inference

# COMMAND ----------

# DBTITLE 1,Grouped Model Prediction Function

predictions = (df
       .groupby("farm_code", "ent_code")
       .applyInPandas(
           dm.grouped_prediction(
            best_models=best_models.toPandas(), 
            feature_cols=["a", "b", "c"], 
            id_cols=["id", "farm_code"]
           ), 
           schema="id int, farm_code string, group_key string, model_url string, prediction float")
       .localCheckpoint()
        ) 
predictions.display()

# COMMAND ----------


