# Databricks notebook source
# MAGIC %md
# MAGIC # Cargar Base de Conocimiento desde el Blog
# MAGIC Scrapea los posts del blog **Spark de Ideas** y los carga en una tabla Delta
# MAGIC para ser indexados por Vector Search.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_bronze")
dbutils.widgets.text("schema", "labs")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

BLOG_URL = "https://mauroloprete.github.io/mauroloprete/blog/"

# COMMAND ----------

# MAGIC %pip install requests beautifulsoup4 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
BLOG_URL = "https://mauroloprete.github.io/mauroloprete/blog/"


def get_post_urls(listing_url: str) -> list[dict]:
    """Extrae las URLs de los posts desde la pagina de listing del blog."""
    resp = requests.get(listing_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    posts = []
    for card in soup.select("#listing-listing .g-col-1"):
        link = card.select_one("a.quarto-grid-link")
        if not link:
            link = card.select_one("a")
        if not link:
            continue

        href = link.get("href", "")
        url = urljoin(listing_url, href)

        title_el = card.select_one("h5.listing-title, .listing-title")
        title = title_el.get_text(strip=True) if title_el else ""

        cats_el = card.select("div.listing-categories .listing-category")
        categories = [c.get_text(strip=True) for c in cats_el]

        if title:
            posts.append({"url": url, "title": title, "categories": categories})

    return posts


def fetch_post_content(url: str) -> str:
    """Descarga un post y extrae el contenido principal como texto limpio."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    article = soup.select_one("main#quarto-document-content") or soup.select_one("main")
    if not article:
        return ""

    for tag in article.select("script, style, nav, .quarto-title-meta, .quarto-appendix"):
        tag.decompose()

    text = article.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Divide un texto largo en chunks respetando lineas."""
    lines = text.split("\n")
    chunks = []
    current = ""

    for line in lines:
        if len(current) + len(line) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = line
        else:
            current = current + "\n" + line if current else line

    if current.strip():
        chunks.append(current.strip())

    return chunks


# COMMAND ----------

table_name = f"{catalog}.{schema}.mauro_docs"

# Migrar tabla si le falta la columna id (primary key del VS index)
if spark.catalog.tableExists(table_name):
    cols = [c.name for c in spark.table(table_name).schema]
    if "id" not in cols:
        print(f"Migrando {table_name}: falta columna 'id', recreando tabla...")
        spark.sql(f"DROP TABLE {table_name}")

# Crear tabla si no existe
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id STRING NOT NULL,
        source STRING NOT NULL,
        title STRING NOT NULL,
        category STRING NOT NULL,
        chunk_id INT NOT NULL,
        content STRING NOT NULL
    )
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# URLs ya cargadas
existing_sources = set(
    row.source for row in spark.sql(f"SELECT DISTINCT source FROM {table_name}").collect()
)

print(f"Scrapeando posts desde {BLOG_URL} ...")
post_list = get_post_urls(BLOG_URL)
print(f"Encontrados {len(post_list)} posts ({len(existing_sources)} ya cargados)")

new_posts = [p for p in post_list if p["url"] not in existing_sources]
print(f"Posts nuevos: {len(new_posts)}")

if not new_posts:
    print("Sin posts nuevos, nada que cargar.")
    dbutils.notebook.exit("Sin posts nuevos")

# COMMAND ----------

documents = []

for post in new_posts:
    print(f"Procesando: {post['title']} ...")
    content = fetch_post_content(post["url"])
    if not content:
        print(f"  (sin contenido, saltando)")
        continue

    chunks = chunk_text(content, max_chars=2000)
    category = post["categories"][0] if post["categories"] else "General"

    for i, chunk in enumerate(chunks):
        documents.append({
            "id": f"{post['url']}#{i}",
            "source": post["url"],
            "title": post["title"],
            "category": category,
            "chunk_id": i,
            "content": chunk,
        })

    print(f"  -> {len(chunks)} chunks")

print(f"\nTotal: {len(documents)} chunks nuevos de {len(new_posts)} posts")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType

table_schema = StructType([
    StructField("id", StringType(), False),
    StructField("source", StringType(), False),
    StructField("title", StringType(), False),
    StructField("category", StringType(), False),
    StructField("chunk_id", IntegerType(), False),
    StructField("content", StringType(), False),
])

df = spark.createDataFrame(documents, schema=table_schema)
df.write.mode("append").saveAsTable(table_name)

total = spark.sql(f"SELECT count(*) FROM {table_name}").collect()[0][0]
print(f"Agregados {len(documents)} chunks nuevos. Total en tabla: {total}")
