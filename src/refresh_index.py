# Databricks notebook source
# MAGIC %md
# MAGIC # Refresh Vector Search Index
# MAGIC Sincroniza la tabla de documentos con el índice de Vector Search.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_bronze")
dbutils.widgets.text("schema", "labs")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

index_name = f"{catalog}.{schema}.soporte_bot_vs_index"
w.vector_search_indexes.sync_index(index_name=index_name)

print(f"Index sync triggered: {index_name}")
