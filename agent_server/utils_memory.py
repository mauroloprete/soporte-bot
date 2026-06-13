import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore

_checkpointer = None
_store = None


@dataclass(frozen=True)
class LakebaseConfig:
    autoscaling_endpoint: Optional[str] = None
    memory_schema: Optional[str] = None

    @property
    def description(self) -> str:
        return self.autoscaling_endpoint or "not configured"


def init_lakebase_config() -> LakebaseConfig:
    endpoint = os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT")
    schema = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA")
    return LakebaseConfig(autoscaling_endpoint=endpoint, memory_schema=schema)


def set_lakebase_resources(checkpointer, store):
    global _checkpointer, _store
    _checkpointer = checkpointer
    _store = store


def get_lakebase_resources():
    return _checkpointer, _store


@asynccontextmanager
async def lakebase_context(config: LakebaseConfig):
    kwargs = {"autoscaling_endpoint": config.autoscaling_endpoint}
    if config.memory_schema:
        kwargs["schema"] = config.memory_schema
    async with AsyncCheckpointSaver(**kwargs) as checkpointer, AsyncDatabricksStore(
        **kwargs
    ) as store:
        yield checkpointer, store
