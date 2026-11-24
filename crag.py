from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS   
from langgraph.graph import END, START, StateGraph, add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_tavily import TavilySearch


load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)


class GradeDocument(BaseModel):
  is_relevant: bool = Field(
      description=(
          "True if the document contains information or keywords relevant to"
          " answering the user's question, otherwise False."
      )
  )


grader_llm = llm.with_structured_output(GradeDocument)

document_loader = DirectoryLoader("./books", glob="*.pdf", loader_cls=PyPDFLoader)
docs = document_loader.load()

sample_docs = docs[:3]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
sample_chunks = text_splitter.split_documents(sample_docs)
print(f"Total sample chunks: {len(sample_chunks)}")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vector_store = FAISS.from_documents(
    documents=sample_chunks, embedding=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("FAISS vector store and retriever are ready")


class CRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list
    filtered_docs: list
    verdict: Literal["correct", "incorrect", "ambiguous"]
    web_search: bool
    generation: str
    
    

def retrieve(state: CRAGState):
  question = state["messages"][-1].content
  docs = retriever.invoke(question)
  return {"question": question, "documents": docs}
    

def evaluate(state: CRAGState):
  question = state["question"]
  documents = state["documents"]

  filtered_docs = []
  for doc in documents:
    prompt = (
        f"User Question: {question}\n\n"
        f"Retrieved Chunk:\n{doc.page_content}\n\n"
        "Does this chunk help answer the question?"
    )
    grade = grader_llm.invoke(prompt)
    if grade.is_relevant:
      filtered_docs.append(doc)

  # Verdict & web_search decision
  if len(filtered_docs) == len(documents) and len(documents) > 0:
    verdict = "correct"
    web_search = False
  elif len(filtered_docs) > 0:
    verdict = "ambiguous"
    web_search = False
  else:
    verdict = "incorrect"
    web_search = True

  return {
      "filtered_docs": filtered_docs,
      "verdict": verdict,
      "web_search": web_search,
  }
  



web_search_tool = TavilySearch(max_results=3)


def web_search(state: CRAGState):
  question = state["question"]

 
  search_results = web_search_tool.invoke({"query": question})

  
  web_docs = []
  results_list = (
      search_results.get("results", [])
      if isinstance(search_results, dict)
      else search_results
  )

  for item in results_list:
    if isinstance(item, dict):
      content = item.get("content", "")
      url = item.get("url", "")
      if content:
        web_docs.append(
            Document(page_content=content, metadata={"source": url})
        )
    elif isinstance(item, str) and item.strip():
      web_docs.append(
          Document(page_content=item, metadata={"source": "web_search"})
      )

  
  return {
      "documents": web_docs,
      "filtered_docs": web_docs,
      "web_search": True,
  }
  
  

def extract_text(content) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for block in content:
      if isinstance(block, dict) and block.get("type") == "text":
        parts.append(block.get("text", ""))
      elif isinstance(block, str):
        parts.append(block)
    return "\n".join(parts).strip()
  return str(content)


def generate(state: CRAGState):
  question = state["question"]
  docs = state.get("filtered_docs") or state.get("documents") or []

  context = "\n\n".join(
      [doc.page_content for doc in docs if getattr(doc, "page_content", None)]
  )

  prompt = (
      "Answer the user's question clearly and accurately using ONLY the"
      " provided context.\n"
      "If the context does not contain enough information to answer, say so"
      " clearly instead of making things up.\n\n"
      f"Context:\n{context}\n\n"
      f"Question:\n{question}"
  )

  response = llm.invoke(prompt)
  answer = extract_text(response.content)  # <-- yahan lagaya

  return {"generation": answer, "messages": [AIMessage(content=answer)]}



def route_after_evaluate(state: CRAGState):
  verdict = state.get("verdict", "incorrect")
  if verdict in ("correct", "ambiguous"):
    return "generate"
  return "web_search"


workflow = StateGraph(CRAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("evaluate", evaluate)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    route_after_evaluate,
    {
        "generate": "generate",
        "web_search": "web_search",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
print("CRAG Graph successfully compiled!")




initial_input = {
    "messages": [
        HumanMessage(
            content="What is the main topic covered in this guide?"
        )
    ]
}

result = app.invoke(initial_input)
print("Verdict:", result.get("verdict"))
print("Web Search Used:", result.get("web_search"))
print("\nAnswer:\n", result.get("generation"))