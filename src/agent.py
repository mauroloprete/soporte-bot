import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.vector_search import VectorSearchClient
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, MessagesState, StateGraph

# --- Retriever ---
vs_client = VectorSearchClient()
vs_index = vs_client.get_index("soporte_bot_vs_index")

# --- LLM ---
w = WorkspaceClient()
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

SYSTEM_PROMPT = (
    "Sos un asistente de soporte técnico. "
    "Respondé usando SOLO el contexto proporcionado. "
    "Si no sabés, decí que no encontrás la información."
)


# --- Graph nodes ---
def retrieve(state: MessagesState):
    question = state["messages"][-1].content
    results = vs_index.similarity_search(
        query_text=question, columns=["content", "source"], num_results=5
    )
    docs = "\n---\n".join(r["content"] for r in results["result"]["data_array"])
    return {"context": docs}


def generate(state: MessagesState):
    context = state.get("context", "")
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContexto:\n{context}"},
        *[{"role": m.type, "content": m.content} for m in state["messages"]],
    ]
    response = w.serving_endpoints.query(name=LLM_ENDPOINT, messages=messages)
    return {"messages": [response.choices[0].message.content]}


# --- Graph ---
graph = StateGraph(MessagesState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# --- Memory (Lakebase) ---
LAKEBASE_URL = os.getenv(
    "LAKEBASE_URL", "postgresql://soporte-bot-mem:5432/postgres"
)
checkpointer = PostgresSaver.from_conn_string(LAKEBASE_URL)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)


# --- MLflow wrapper ---
class SoporteBot(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        question = model_input["question"][0]
        thread_id = model_input.get("thread_id", ["default"])[0]

        result = app.invoke(
            {"messages": [("user", question)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["messages"][-1].content
