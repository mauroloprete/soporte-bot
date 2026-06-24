# Spark de Ideas Bot

Réplica de [Mauro Loprete](https://mauroloprete.github.io/mauroloprete/) para buenas prácticas de Databricks. Agente RAG con memoria conversacional, deployado con **Declarative Automation Bundles (DABs)**.

Ejemplo de la charla *"De YAML a producción"* — Databricks Meetup Uruguay, junio 2026.

## Arquitectura

```
Blog (HTML) → Load KB Job → Delta Table → Vector Search
                                                ↑ retrieval
Usuario → Databricks App (FastAPI + chat.html) ─┘
               ↓ generate           ↓ checkpoints     ↓ tracing
          AI Gateway V2         Lakebase (memoria)  MLflow Experiment
          (jailbreak guardrail)
               ↓
          AI Gateway legacy
          (rate limits + PII + safety)
               ↓
          Foundation Model (Llama 3.3 70B)
```

**Un solo `databricks.yml` despliega 6 recursos:**

| Recurso | Tipo DABs |
|---------|-----------|
| MLflow Experiment | `experiments` |
| Vector Search Endpoint | `vector_search_endpoints` |
| Model Serving Endpoint (AI Gateway) | `model_serving_endpoints` |
| Lakebase PostgreSQL (memoria) | `apps.resources.postgres` |
| Databricks App (FastAPI) | `apps` |
| Job (load KB + refresh index) | `jobs` |

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

## AI Gateway V2 — guardrails LLM-based

La app usa dos capas de guardrails:

- **Legacy (DABs):** `safety: true`, `pii: BLOCK` — reglas estáticas declaradas en `databricks.yml` sobre el Model Serving Endpoint.
- **AI Gateway V2 (Beta):** guardrails evaluados por Gemma 3 12B — jailbreak y hallucination. Se configuran desde la UI de AI Gateway.

Para que la App acceda al endpoint V2, se usa **on-behalf-of-user auth**:

```yaml
# databricks.yml
apps:
  mauro_bot_app:
    user_api_scopes:
      - ai-gateway          # Habilita on-behalf-of-user para V2
```

El middleware en `start_server.py` captura `x-forwarded-access-token` del header HTTP y lo pasa al LangGraph state. En `generate`, el token del usuario se usa como `api_key` del OpenAI client apuntando a `/ai-gateway/mlflow/v1`.

## Demo: dos apps, mismo código

| App | LLM Endpoint | Guardrails | Jailbreak |
|-----|--------------|------------|-----------|
| `mauro-bot` | AI Gateway V2 (`mauro-bot-llm-endpoint`) | Jailbreak + PII + safety | BLOQUEADO |
| `mauro-bot-no-gw` | Serving endpoint (`mauro-bot-llm-gateway`) | Solo PII + safety | Vulnerable |

La diferencia se controla con dos env vars: `USE_AI_GATEWAY` y `LLM_ENDPOINT`.

## Estructura

```
mauro-bot/
├── databricks.yml              # Todo el deploy (6 recursos)
├── app.yaml                    # Runtime config (command + env)
├── pyproject.toml              # Deps (uv)
├── agent_server/
│   ├── agent.py                # LangGraph + DatabricksOpenAI + guardrails
│   ├── start_server.py         # FastAPI + Lakebase init + UserTokenMiddleware
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
| 5 | AI Gateway V2 no tiene recurso DABs | Los endpoints V2 se crean desde la UI, no desde `databricks.yml`. |
| 6 | V2 requiere `user_api_scopes: [ai-gateway]` | Sin esto, el SP de la app no puede acceder a endpoints V2. On-behalf-of-user es obligatorio. |
| 7 | Python 3.11 no propaga `contextvars` a threads | LangGraph usa `run_in_executor`. Pasar el user token via state, no via `ContextVar`. |

## CI/CD

El workflow de GitHub Actions tiene dos jobs:

- **PR**: `bundle validate` + `bundle deploy --compute-id=auto` (comenta el plan en el PR)
- **Merge a main**: `bundle deploy -t prod`

Secretos necesarios en GitHub:
- `DATABRICKS_HOST`: URL del workspace
- `DATABRICKS_TOKEN`: Token de service principal

## Links

- [Documentación DABs](https://docs.databricks.com/aws/en/dev-tools/bundles)
- [AI Gateway V2 — author agent](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent)
- [AI Gateway — guardrails](https://docs.databricks.com/aws/en/ai-gateway/guardrails)
- [Databricks Apps — auth](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth)
- [Lakebase con DABs](https://learn.microsoft.com/en-us/azure/databricks/oltp/projects/manage-with-bundles)
- [Agent memory con Lakebase](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/stateful-agents)
- [Blog: Spark de Ideas](https://mauroloprete.github.io/mauroloprete/)
