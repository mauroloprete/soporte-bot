import contextvars
import logging
import os
from typing import Any, AsyncGenerator, Sequence, TypedDict

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, StateGraph, add_messages
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from openai import BadRequestError, OpenAI
from typing_extensions import Annotated

from agent_server.utils_memory import get_lakebase_resources

# Token del usuario logueado (on-behalf-of-user auth)
_user_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_user_token", default=None
)

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

VS_INDEX_NAME = os.getenv("VS_INDEX_NAME", "dev_bronze.labs.mauro_bot_vs_index")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
USE_AI_GATEWAY = os.getenv("USE_AI_GATEWAY", "false").lower() == "true"

SYSTEM_PROMPT = (
    "Sos Mauro Bot, una replica de Mauro Loprete — Data Engineer "
    "y autor del blog Spark de Ideas (https://mauroloprete.github.io/mauroloprete/blog/). "
    "Respondés sobre buenas practicas de Databricks. Usá un tono relajado rioplatense "
    "pero sin exagerar — NO empieces cada respuesta con 'Che'.\n\n"
    "Reglas ESTRICTAS:\n"
    "- Respondé UNICAMENTE con informacion que aparezca en el contexto proporcionado.\n"
    "- Si la respuesta NO esta en el contexto, decí: 'No tengo informacion sobre "
    "eso en mi base de conocimiento. Te recomiendo visitar el blog Spark de Ideas "
    "donde Mauro publica contenido sobre Databricks y Data Engineering.'\n"
    "- NUNCA inventes datos, nombres de productos, APIs, endpoints o funcionalidades "
    "que no esten en el contexto.\n"
    "- SIEMPRE mencioná el blog Spark de Ideas como fuente y recomendá leer el post "
    "original para mas detalle.\n"
    "- Incluí la URL de la fuente cuando aparezca en el campo 'source' del contexto.\n"
    "- Usá ejemplos de codigo solo si aparecen en el contexto.\n"
    "- Respondé de forma concisa y directa. No repitas toda la info del contexto, "
    "resumí los puntos clave.\n"
    "- NO uses formato markdown (ni *, ni **, ni ``` ni #). Respondé en texto plano. "
    "Para listas usá guiones simples (-)."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    context: str
    user_token: str | None


_w = None
_llm_client = None


def _get_workspace():
    global _w
    if _w is None:
        _w = WorkspaceClient()
    return _w


def _get_llm_client(user_token: str | None = None):
    """Devuelve el cliente LLM.

    Cuando USE_AI_GATEWAY=True y hay un user token (on-behalf-of-user),
    crea un OpenAI client apuntando a /ai-gateway/mlflow/v1 con el token
    del usuario. Esto permite que los guardrails de AI Gateway V2 apliquen.
    Sin user token, cae al DatabricksOpenAI con credenciales del SP.
    """
    if USE_AI_GATEWAY and user_token:
        host = os.getenv("DATABRICKS_HOST", "")
        return OpenAI(
            api_key=user_token,
            base_url=f"https://{host}/ai-gateway/mlflow/v1",
        )
    global _llm_client
    if _llm_client is None:
        _llm_client = DatabricksOpenAI(use_ai_gateway=False)
    return _llm_client


def retrieve(state: AgentState):
    question = state["messages"][-1].content
    w = _get_workspace()
    resp = w.vector_search_indexes.query_index(
        index_name=VS_INDEX_NAME,
        query_text=question,
        columns=["content", "source"],
        num_results=5,
    )
    rows = resp.result.data_array or []
    docs = "\n---\n".join(
        f"[Fuente: {row[1]}]\n{row[0]}" if len(row) > 1 and row[1] else row[0]
        for row in rows
    )
    return {"context": docs}


def generate(state: AgentState):
    context = state.get("context", "")
    client = _get_llm_client(user_token=state.get("user_token"))
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContexto:\n{context}"},
        *[
            {
                "role": {"human": "user", "ai": "assistant"}.get(m.type, m.type),
                "content": m.content,
            }
            for m in state["messages"]
        ],
    ]
    try:
        resp = client.chat.completions.create(
            model=LLM_ENDPOINT,
            messages=messages,
        )
        text = resp.choices[0].message.content
    except BadRequestError as exc:
        err_msg = str(exc)
        logger.error("BadRequestError from LLM: %s", err_msg)
        if "REQUEST_BLOCKED_BY_GUARDRAIL" in err_msg or "guardrail" in err_msg.lower():
            if "pii_detection\":true" in err_msg.lower() or "pii: block" in err_msg.lower():
                text = (
                    "Tu mensaje fue bloqueado por contener datos personales sensibles "
                    "(como tarjetas de credito o documentos de identidad). "
                    "Por tu seguridad, no puedo procesar ese tipo de informacion."
                )
            elif "Jailbreak" in err_msg or "jailbreak" in err_msg.lower():
                text = (
                    "Tu mensaje fue bloqueado por los guardrails de seguridad "
                    "al detectar un intento de manipulacion. "
                    "Intenta reformular tu pregunta sobre Databricks o Data Engineering."
                )
            else:
                text = (
                    "Tu mensaje fue bloqueado por los guardrails de seguridad. "
                    "Intenta reformular tu pregunta sobre Databricks o Data Engineering."
                )
        else:
            raise
    return {"messages": [AIMessage(content=text)]}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph


async def init_agent(checkpointer=None):
    graph = _build_graph()
    return graph.compile(checkpointer=checkpointer)


def _get_or_create_thread_id(request: ResponsesAgentRequest) -> str:
    ci = dict(request.custom_inputs or {})
    if "thread_id" in ci and ci["thread_id"]:
        return str(ci["thread_id"])
    if request.context and getattr(request.context, "conversation_id", None):
        return str(request.context.conversation_id)
    import uuid_utils

    return str(uuid_utils.uuid7())


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    thread_id = _get_or_create_thread_id(request)
    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    user_token = _user_token.get()
    input_state = {
        "messages": to_chat_completions_input(
            [i.model_dump() for i in request.input]
        ),
        "user_token": user_token,
    }

    checkpointer, _store = get_lakebase_resources()
    agent = await init_agent(checkpointer=checkpointer)
    async for event in _stream_events(agent, input_state, config):
        yield event


async def _stream_events(agent, input_state, config):
    async for event in agent.astream(input_state, config, stream_mode=["updates", "messages"]):
        kind, data = event
        if kind == "messages":
            msg, metadata = data
            if msg.content and metadata.get("langgraph_node") == "generate":
                yield ResponsesAgentStreamEvent(
                    type="response.output_text.delta",
                    item_id="msg_1",
                    output_index=0,
                    content_index=0,
                    delta=msg.content,
                )
        elif kind == "updates" and "generate" in data:
            last_msg = data["generate"]["messages"][-1]
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item_id="msg_1",
                output_index=0,
                item={
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": last_msg.content}],
                },
            )
