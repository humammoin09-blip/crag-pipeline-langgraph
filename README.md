# Corrective RAG (CRAG) with LangGraph & Google Gemini

A production-grade implementation of **Corrective Retrieval-Augmented Generation (CRAG)** built with **LangGraph**, **Google Gemini**, **FAISS**, and **Tavily Search**.

This project implements an agentic self-correcting retrieval workflow designed to eliminate hallucinations, filter out noisy document chunks, and dynamically trigger external web search when local knowledge retrieval is insufficient.

---

## Overview: The Problem with Standard RAG

In standard RAG (Retrieval-Augmented Generation) pipelines, the system follows a rigid, one-way path:
1. Embed the query.
2. Retrieve the top-$k$ nearest chunks from a vector store.
3. Feed the retrieved chunks into an LLM prompt as context.

### Where Standard RAG Fails
- **Irrelevant Context Injection:** Vector similarity does not guarantee semantic relevance. Chunks matching surface keywords can mislead the LLM.
- **Hallucination under Ignorance:** If the internal corpus lacks the answer, the LLM often attempts to extrapolate from poor context rather than seeking new knowledge.
- **No Self-Correction:** Standard RAG pipelines lack an internal feedback loop to evaluate retrieval quality before generation.

### The CRAG Solution
**Corrective RAG (CRAG)** introduces an evaluation and self-correction layer between retrieval and generation:
- An LLM-powered **Document Grader** inspects each retrieved chunk against the query.
- If relevant context exists, the workflow filters out noise and proceeds to **grounded generation**.
- If the retrieved context is irrelevant or ambiguous, the workflow dynamically branches to a **Web Search Fallback (Tavily)** to retrieve authoritative external context before synthesizing the final answer.

---

## Architecture & Workflow

The pipeline is modeled as a state machine using **LangGraph** (`StateGraph`), providing deterministic state management, inspectable intermediate steps, and conditional branching.

### State Graph Flow

```
                      +-----------------------+
                      |         START         |
                      +-----------+-----------+
                                  |
                                  v
                      +-----------------------+
                      |        retrieve       |  <-- FAISS Vector Store
                      +-----------+-----------+
                                  |
                                  v
                      +-----------------------+
                      |        evaluate       |  <-- LLM Document Grader (Pydantic)
                      +-----------+-----------+
                                  |
                        [decide_to_generate]
                            /           \
       (verdict == "correct")            (web_search == True)
                          /                 \
                         v                   v
               +-----------------+   +-----------------+
               |    generate     |   |   web_search    |  <-- Tavily Search API
               +--------+--------+   +--------+--------+
                        |                     |
                        |                     v
                        |            +-----------------+
                        |            |    generate     |  <-- Gemini LLM
                        |            +--------+--------+
                        |                     |
                        +----------+----------+
                                   |
                                   v
                              +----+----+
                              |   END   |
                              +---------+
```

---

### State Schema (`CRAGState`)

The graph passes a typed dictionary (`CRAGState`) across all nodes:

```python
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class CRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Chat history with append reducer
    question: str                                         # Extracted user question
    documents: list                                       # Raw chunks from vector store
    filtered_docs: list                                  # Graded & filtered relevant chunks
    verdict: Literal["correct", "incorrect", "ambiguous"] # Assessment verdict
    web_search: bool                                      # Route flag for web fallback
    generation: str                                       # Final synthesized answer
```

---

### Node Breakdown

| Node | Purpose | Implementation Details |
| :--- | :--- | :--- |
| **`retrieve`** | Fetches the top-$k$ context chunks from the local vector database. | Extracts user query from `state["messages"]`, queries FAISS index (`as_retriever(search_kwargs={"k": 3})`), and populates `state["documents"]`. |
| **`evaluate`** | Grades each chunk's relevance to the question and assigns a verdict. | Uses Gemini structured output via a Pydantic schema (`GradeDocument`). Keeps relevant chunks in `filtered_docs`. Sets `verdict="correct"` if sufficient context exists; sets `verdict="incorrect"` and `web_search=True` otherwise. |
| **`decide_to_generate`** *(Edge)* | Determines graph routing based on retrieval evaluation. | Conditional edge function inspecting `state["web_search"]`. Routes to `generate` if `False`, or branches to `web_search` if `True`. |
| **`web_search`** | Gathers external information when local documents fail. | Executes a query via Tavily Search, parses external snippets, normalizes results into LangChain `Document` objects, and appends them to context. |
| **`generate`** | Produces the final grounded answer. | Formats prompt containing validated context, invokes `ChatGoogleGenerativeAI`, normalizes output text blocks, and appends the response to `messages`. |

---

## Tech Stack

- **Orchestration & State Machine:** [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph`, `START`, `END`, conditional edges, state reducers)
- **Large Language Model:** [Google Gemini](https://ai.google.dev/) (`gemini-2.5-flash` / `gemini-1.5-flash`) via `langchain-google-genai`
- **Embeddings:** `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2` / `text-embedding-004`)
- **Vector Store:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) (`faiss-cpu`)
- **Document Loading & Chunking:** `pypdf`, `DirectoryLoader`, and `RecursiveCharacterTextSplitter`
- **Web Search Engine:** [Tavily](https://tavily.com/) via `langchain-tavily`
- **Data Validation & Typing:** [Pydantic v2](https://docs.pydantic.dev/) (`BaseModel`, `Field`) & Python `TypedDict`
- **Environment & Configuration:** `python-dotenv`

---

## Project Structure

```
CRAG/
├── books/                     # Local PDF corpus directory (gitignored)
├── .env.example               # Template for required API keys
├── .gitignore                 # Excludes .venv, .env, PDFs, and indices
├── crag_project.ipynb         # End-to-end interactive notebook implementation
├── requirements.txt           # Python dependency specifications
└── README.md                  # Project documentation & architecture guide
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))
- Tavily API Key ([Tavily AI](https://tavily.com/))

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/CRAG.git
cd CRAG
```

### 3. Create and Activate a Virtual Environment
**On Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to create your `.env` file:
```bash
cp .env.example .env
```
Open `.env` and fill in your API credentials:
```env
GOOGLE_API_KEY="your-google-api-key-here"
TAVILY_API_KEY="your-tavily-api-key-here"
```

### 6. Prepare Document Corpus
Place your reference PDF documents into the `./books` directory:
```bash
mkdir -p books
# Copy sample PDF files into the books folder
```

---

## Usage & Execution

Launch Jupyter Lab or open `crag_project.ipynb` in your preferred IDE:

```bash
jupyter lab
# or: code crag_project.ipynb
```

### Invocation Code Example

```python
from langchain_core.messages import HumanMessage
from crag_project import app  # Compiled LangGraph instance

# Define user query
input_state = {
    "messages": [HumanMessage(content="What are the core stages of the CRAG workflow?")]
}

# Run the state machine
result = app.invoke(input_state)

# Inspect outcomes
print("Verdict:   ", result["verdict"])
print("Web Search:", result["web_search"])
print("Answer:    \n", result["generation"])
```

### Example Outputs

#### Scenario A: In-Domain Query (Resolved from Local Vector Store)
```json
{
  "question": "What is the primary topic discussed in chapter 2?",
  "verdict": "correct",
  "web_search": false,
  "filtered_docs": 2,
  "generation": "Chapter 2 focuses on recursive character chunking and semantic overlap strategies..."
}
```
*Flow: `START` -> `retrieve` -> `evaluate` -> `generate` -> `END`*

#### Scenario B: Out-of-Domain Query (Automatic Web Search Fallback)
```json
{
  "question": "What were the major generative AI announcements at Google I/O this year?",
  "verdict": "incorrect",
  "web_search": true,
  "filtered_docs": 0,
  "generation": "Based on real-time web search results, Google announced several major updates including..."
}
```
*Flow: `START` -> `retrieve` -> `evaluate` -> `web_search` -> `generate` -> `END`*

---

## Key Learnings & Engineering Takeaways

1. **Deterministic Grading with Structured Output:**
   Prompting an LLM for free-form relevance grading leads to parsing errors (e.g., lowercase vs. uppercase, explanations instead of boolean flags). Using Pydantic models with `llm.with_structured_output(GradeDocument)` guarantees deterministic boolean checks (`is_relevant: bool`), eliminating regex brittleness.

2. **Handling Rate Limits on Embedding Batches:**
   Large PDF documents generate hundreds of chunks. Submitting all chunks in a single call to `GoogleGenerativeAIEmbeddings` can trigger API rate limits (`429 Too Many Requests`). Batching embedding creation in chunks of 50-100 prevents network timeouts and respects quota tiers.

3. **Gemini Output Normalization:**
   Depending on version and task modality, Gemini models may return content as structured content blocks (`list[dict]`) or string attributes. Implementing a helper that safely extracts string values (`getattr(msg, "content", str(msg))`) prevents downstream type errors in LangGraph state propagation.

4. **Preserving Message Reducer Semantics:**
   Using LangGraph's `add_messages` annotation ensures conversation history is preserved incrementally across node transitions without manually appending lists or risking state overwrite.

---

## Future Improvements

- [ ] **Query Rewriting Node:** Introduce a query transformation step before Tavily search to optimize search keywords when the original user query is conversational or vague.
- [ ] **Multi-Vector & Hybrid Search:** Combine BM25 keyword search with FAISS dense retrieval using an ensemble retriever for higher recall.
- [ ] **Reranking:** Incorporate a cross-encoder reranker (e.g., Cohere Rerank or BGE Reranker) before the grading node to minimize LLM grading tokens.
- [ ] **RAG Evaluation Suite:** Implement automated evaluation metrics (faithfulness, answer relevancy, context precision) using Ragas or TruLens.

---

## License

This project is licensed under the MIT License - feel free to adapt and build upon it.
