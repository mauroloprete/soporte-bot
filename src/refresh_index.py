# Databricks notebook source
# MAGIC %md
# MAGIC # Refresh Vector Search Index
# MAGIC Crea (si no existe) y sincroniza el indice de Vector Search
# MAGIC con la tabla de documentos.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_bronze")
dbutils.widgets.text("schema", "labs")
dbutils.widgets.text("vs_endpoint_name", "mauro-bot-vs")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)

w = WorkspaceClient()

index_name = f"{catalog}.{schema}.mauro_bot_vs_index"
source_table = f"{catalog}.{schema}.mauro_docs"
vs_endpoint = dbutils.widgets.get("vs_endpoint_name")

# Habilitar Change Data Feed (requerido por Vector Search)
spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"CDF habilitado en {source_table}")

# Verificar si el indice ya existe
existing = [idx.name for idx in w.vector_search_indexes.list_indexes(endpoint_name=vs_endpoint)]

if index_name in existing:
    import time

    idx_info = w.vector_search_indexes.get_index(index_name=index_name)
    status = idx_info.status.ready
    print(f"Index {index_name} existe (ready={status})")

    if not status:
        print("Index no esta listo, esperando...")
        for _ in range(30):
            time.sleep(30)
            idx_info = w.vector_search_indexes.get_index(index_name=index_name)
            if idx_info.status.ready:
                print("Index listo!")
                break
            print(f"  Todavia inicializando... esperando 30s mas")
        else:
            print("Index no se puso ready en 15 min, saltando sync")

    if idx_info.status.ready:
        print(f"Sincronizando index...")
        w.vector_search_indexes.sync_index(index_name=index_name)
    else:
        print("Saltando sync, el index se sincronizara automaticamente al estar listo")
else:
    print(f"Index {index_name} no existe, creando...")
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=vs_endpoint,
        primary_key="id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name="databricks-gte-large-en",
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
        ),
    )

print(f"Index sync triggered: {index_name}")
