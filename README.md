# Spark de Ideas Bot

Réplica de [Mauro Loprete](https://mauroloprete.github.io/mauroloprete/) para buenas prácticas de Databricks. Agente RAG con memoria conversacional, deployado con **Declarative Automation Bundles (DABs)**.

Ejemplo de la charla *"De YAML a producción"* — Databricks Meetup Uruguay, junio 2026.

## Arquitectura

```
Blog (HTML) → Load KB Job → Delta Table → Vector Search
                                                ↑ retrieval
Usuario → Databricks App (FastAPI + chat.html) ─┘
               ↓ checkpoints        ↓ tracing
          Lakebase (memoria)    MLflow Experiment
```

**Un solo `databricks.yml` despliega 5 recursos:**

| Recurso | Tipo DABs |
|---------|-----------|
| MLflow Experiment | `experiments` |
| Vector Search Endpoint | `vector_search_endpoints` |
| Lakebase PostgreSQL (memoria) | `postgres_projects` |
| Databricks App (FastAPI) | `apps` |
| Job (load KB + refresh index) | `jobs` |

El agente corre directo en la App — sin Model Serving Endpoint ni Registered Model.

## Requisitos

- Databricks CLI >= 0.230
- Workspace con Unity Catalog, Vector Search y Lakebase habilitados

## Setup rápido

1. Cloná el repo:
   ```bash
   git clone <url>
   cd mauro-bot
   ```

2. Configurá autenticación con el CLI:
   ```bash
   databricks auth login --host https://TU-WORKSPACE.azuredatabricks.net
   ```

3. (Opcional) Si tu workspace usa otros catálogos, editá las variables en `databricks.yml`:
   ```yaml
   variables:
     catalog:
       default: tu_catalogo
     schema:
       default: tu_schema
   ```

4. Validá y deployá:
   ```bash
   databricks bundle validate -t dev
   databricks bundle deploy -t dev
   ```

5. Cargá la base de conocimiento:
   ```bash
   databricks bundle run load_knowledge_base -t dev
   ```
   El job scrapea el blog, genera chunks y sincroniza el Vector Search index.

6. Abrí la Databricks App — el chat está en la raíz.

## Base de conocimiento

El job `load_knowledge_base` scrapea directamente el blog [Spark de Ideas](https://mauroloprete.github.io/mauroloprete/blog/):

1. Fetch de la listing page con `requests`
2. Extrae URLs de todos los posts con `BeautifulSoup`
3. Descarga cada post, limpia HTML, extrae texto
4. Genera chunks de ~2000 caracteres con título, categoría y URL fuente
5. Escribe a tabla Delta en Unity Catalog

Si se publica un post nuevo, el próximo run lo levanta automáticamente.

## Estructura

```
mauro-bot/
├── databricks.yml              # Todo el deploy (5 recursos)
├── app.yaml                    # Runtime config (command + env)
├── pyproject.toml              # Deps (uv)
├── agent_server/
│   ├── agent.py                # LangGraph + ResponsesAgent
│   ├── start_server.py         # FastAPI + Lakebase init
│   ├── utils_memory.py         # CheckpointSaver + Store
│   └── chat.html               # Chat UI (streaming SSE + marked.js)
├── src/
│   ├── load_knowledge_base.py  # Scraping del blog → Delta table
│   └── refresh_index.py        # Sync Vector Search index
└── .github/
    └── workflows/
        └── deploy.yml          # CI/CD con GitHub Actions
```

## Gotchas

| # | Gotcha | Detalle |
|---|--------|---------|
| 1 | `postgres_projects` siempre POST | Bug [#5183](https://github.com/databricks/cli/issues/5183) — falla con "already exists" en re-deploy. Crear via UI. |
| 2 | Permisos Lakebase | SP con `CAN_CONNECT_AND_CREATE` no crea tablas en `public`. Requiere grants como superuser. |
| 3 | `valueFrom` vs `value_from` | `app.yaml` usa camelCase (`valueFrom`), `databricks.yml` usa snake_case (`value_from`). |
| 4 | `apps stop/start` pierde config | Recrear deployment manual no hereda la config del bundle. Usar `bundle deploy`. |

## CI/CD

El workflow de GitHub Actions tiene dos jobs:

- **PR**: `bundle validate` + `bundle deploy --compute-id=auto` (comenta el plan en el PR)
- **Merge a main**: `bundle deploy -t prod`

Secretos necesarios en GitHub:
- `DATABRICKS_HOST`: URL del workspace
- `DATABRICKS_TOKEN`: Token de service principal

## Links

- [Documentación DABs](https://docs.databricks.com/aws/en/dev-tools/bundles)
- [Lakebase con DABs](https://learn.microsoft.com/en-us/azure/databricks/oltp/projects/manage-with-bundles)
- [Agent memory con Lakebase](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/stateful-agents)
- [Blog: Spark de Ideas](https://mauroloprete.github.io/mauroloprete/)
