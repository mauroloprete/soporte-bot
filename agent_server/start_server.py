"""Agent server entry point. load_dotenv must run before agent imports (auth config)."""

# ruff: noqa: E402
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi.responses import HTMLResponse
from databricks_ai_bridge.long_running import LongRunningAgentServer
from mlflow.genai.agent_server import setup_mlflow_git_based_version_tracking

import os

from agent_server.agent import _user_token
from agent_server.utils_memory import init_lakebase_config, lakebase_context, set_lakebase_resources

logger = logging.getLogger(__name__)

import agent_server.agent  # noqa: F401

LAKEBASE_CONFIG = init_lakebase_config()

agent_server = LongRunningAgentServer(
    "ResponsesAgent",
    enable_chat_proxy=False,
    db_autoscaling_endpoint=LAKEBASE_CONFIG.autoscaling_endpoint,
)

app = agent_server.app


class UserTokenMiddleware(BaseHTTPMiddleware):
    """Captura x-forwarded-access-token y lo inyecta en el ContextVar."""

    async def dispatch(self, request: Request, call_next):
        token = request.headers.get("x-forwarded-access-token")
        tok = _user_token.set(token)
        try:
            return await call_next(request)
        finally:
            _user_token.reset(tok)


app.add_middleware(UserTokenMiddleware)

try:
    setup_mlflow_git_based_version_tracking()
except Exception as e:
    logger.warning("MLflow git version tracking unavailable: %s", e)

CHAT_HTML = (Path(__file__).parent / "chat.html").read_text()
_USE_AI_GATEWAY = os.environ.get("USE_AI_GATEWAY", "false").lower() == "true"


@app.get("/", response_class=HTMLResponse)
async def chat_page():
    badge = "AI Gateway V2" if _USE_AI_GATEWAY else "Sin AI Gateway"
    color = "#0d9488" if _USE_AI_GATEWAY else "#dc2626"
    return CHAT_HTML.replace("<!--GATEWAY_BADGE-->", f'<span style="background:{color};padding:.2rem .6rem;border-radius:1rem;font-size:.75rem;font-weight:600;">{badge}</span>')


_original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _lifespan(app):
    if not LAKEBASE_CONFIG.autoscaling_endpoint:
        logger.warning("Lakebase not configured — memory disabled")
        try:
            async with _original_lifespan(app):
                yield
        except Exception:
            yield
        return

    try:
        async with lakebase_context(LAKEBASE_CONFIG) as (checkpointer, store):
            await checkpointer.setup()
            await store.setup()
            logger.info("Lakebase memory initialized")
            set_lakebase_resources(checkpointer, store)

            try:
                async with _original_lifespan(app):
                    yield
            except Exception as exc:
                logger.warning("Long-running DB init failed: %s. Background mode disabled.", exc)
                yield
    except Exception as exc:
        logger.error("Lakebase setup failed: %s — starting without memory", exc)
        yield


app.router.lifespan_context = _lifespan


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
