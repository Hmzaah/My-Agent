# How Jolt works

A walkthrough of every part of the system, why it is built that way, and the questions an
interviewer is likely to ask about it.

## 1. The problem

A language model only knows what it saw in training, and it will confidently make things up
when it does not know. Retrieval-augmented generation (RAG) fixes part of this: find relevant
passages in your own documents, put them in the prompt, and ask the model to answer from them.

Plain RAG still fails in three common ways:

1. **Bad retrieval.** The question uses different words than the document, so search misses the
   right passage, and the model answers from irrelevant text.
2. **No answer in the index.** The documents simply do not contain the answer, and the model
   fills the gap with a guess.
3. **No accountability.** You cannot tell which part of the answer came from where.

Jolt's agent loop targets each one: grading and query rewriting for (1), web fallback and
abstention for (2), numbered citations for (3).

## 2. Ingestion (`ingest.py`, `store.py`)

**Loading.** PDFs are read page by page with `pypdf`, so every chunk remembers its page number.
Text and Markdown files are read whole.

**Chunking.** Text is split recursively: first on blank lines (paragraphs), then single lines,
then sentences, then words, only going finer when a piece is still too long. Pieces are packed
into chunks of about 800 characters. Each chunk starts with the last ~120 characters of the
previous one (the *overlap*) so a fact that straddles a boundary still appears whole in one
chunk.

*Why recursive splitting?* Fixed-size splitting cuts sentences in half. Splitting on natural
boundaries first keeps chunks readable, which helps both the embedding model and the LLM.

**Chunk IDs** are a hash of source, page, position and text. Re-ingesting the same file does not
create duplicates.

**Two indexes over the same chunks:**

- **Dense (FAISS).** Each chunk becomes a 384-number vector from `all-MiniLM-L6-v2`. Vectors are
  normalised to length 1, so the inner product equals cosine similarity, and an exact
  `IndexFlatIP` search finds the closest chunks. Good at meaning and paraphrase.
- **Sparse (BM25).** Classic keyword scoring: rewards chunks that contain the query's words,
  especially rare ones, with diminishing returns for repeats and a penalty for long chunks.
  Good at exact names, acronyms and part numbers that embeddings blur.

The index, chunks and metadata are saved to `data/index/`. On load, Jolt checks the saved
embedding model matches the current one, because mixing vectors from two models gives
meaningless search results.

## 3. Retrieval (`retrieval.py`)

1. Take the top 20 from FAISS and the top 20 from BM25.
2. Merge them with **reciprocal rank fusion (RRF)**: each chunk scores `sum(1 / (60 + rank))`
   over the lists it appears in. RRF only uses positions, so it can combine a cosine score and
   a BM25 score without normalising them. A chunk ranked well by both rises to the top.
3. **Rerank** the merged candidates with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) and keep the
   top 5.

*Why a cross-encoder?* The embedding model encodes query and chunk separately, then compares
vectors: fast, but it never sees them together. A cross-encoder reads the pair as one input and
scores relevance directly: more accurate, but too slow to run on every chunk. So it is used only
on the 20 or so candidates from the cheap first stage. This two-stage "retrieve then rerank"
pattern is standard in search systems.

## 4. The agent (`agent/`)

The agent is a **LangGraph** `StateGraph`. The *state* is a dictionary (question, current query,
retrieved chunks, number of rewrites, answer, trace). Each *node* is a function that reads the
state and returns only the fields it changes. *Edges* decide which node runs next.

| Node | What it does |
|---|---|
| `plan` | Asks the LLM to turn the question into a focused search query (keeps technical terms, drops filler). Falls back to the raw question if the model's output is not valid JSON. |
| `retrieve` | Runs hybrid retrieval with the current query. |
| `grade` | Shows the LLM the numbered chunks and asks which are relevant and whether together they answer the question. Irrelevant chunks are dropped. |
| `rewrite` | If the evidence was insufficient, asks the LLM for a different query (synonyms, expanded acronyms, the missing piece) and loops back to `retrieve`. |
| `web_search` | If rewriting is exhausted (or the index is empty), searches the web and sends the results back through `grade`. |
| `generate` | Asks the LLM to answer only from the numbered sources, citing them as `[1]`, `[2]`, or to reply "I don't know based on the available sources." |

**Routing after `grade`** (in `graph.py`):

1. Evidence sufficient → `generate`.
2. Else, if there are rewrites left and the index is not empty → `rewrite`.
3. Else, if web search is allowed and not yet used → `web_search`.
4. Else → `generate`, which will usually abstain.

Loops are bounded: at most `max_rewrites` rewrites and one web search, and the graph's
recursion limit is set from those numbers, so a question can never spin forever.

**Structured outputs.** The planner, grader and rewriter must reply in JSON. Both backends request
JSON mode from the model server. Because small models still sometimes add prose or code fences,
`parse_json` extracts the first JSON object, and every node has a safe fallback when parsing
fails (e.g. the grader keeps all chunks and lets the answer step decide).

**Citations.** The answer is scanned for `[n]` markers. Only numbers that match a source become
citations, so a model cannot cite a source that does not exist. `grounded` is true only when the
answer cites at least one source.

**Trace.** Every node appends a step with its timing, so the API and UI can show exactly what
the agent did: useful for debugging and for trusting the answer.

## 5. Model backends (`llm.py`, `embeddings.py`)

Every node talks to an `LLM` protocol with one method, `chat(messages, ...)`.

- **`LlamaCppLLM`** runs a quantised GGUF model locally with `llama-cpp-python`, offloading
  layers to the GPU. It uses the model's own chat template, so switching from Qwen to Phi-3 or
  Llama needs no prompt changes. A lock serialises calls, because one llama.cpp context cannot
  run two generations at once.
- **`OpenAICompatibleLLM`** works with anything that speaks the OpenAI API: Ollama, vLLM,
  LM Studio, Groq, OpenAI. If a server rejects JSON mode, it retries once without it and
  remembers that.

*Why quantisation?* A 3B-parameter model in 16-bit needs about 6 GB of memory; 4-bit
quantisation (Q4_K_M) cuts that to about 2 GB with a small quality loss, which is what makes it
fit on a 4 GB laptop GPU alongside the embedding model.

## 6. Serving (`api.py`, `cli.py`, `ui/`)

- **FastAPI**: `POST /v1/ask`, `POST /v1/documents` (upload), `GET /health`, `GET /v1/graph`.
  Requests and responses are Pydantic models, so invalid input gets a 422 automatically and the
  OpenAPI docs at `/docs` are generated. Model inference is blocking, so `/v1/ask` runs it in a
  thread pool instead of blocking the event loop. Every response carries an `x-request-id`
  header that also appears in the logs.
- **CLI** (Typer): `jolt ingest | ask | serve | eval | graph`.
- **Streamlit UI** calls the API rather than loading models itself, so the model lives in one
  process.
- **Docker Compose** runs Ollama, the API and the UI together.

## 7. Evaluation (`evaluation.py`)

You cannot improve what you do not measure. `jolt eval` runs a labelled question set and reports:

- **hit@k**: was a correct source in the top k chunks? (retrieval recall)
- **MRR**: mean of 1 / rank of the first correct chunk (rewards ranking it first)
- **answer rate / abstention accuracy**: does it answer answerable questions and refuse
  unanswerable ones?
- **keyword recall**: do expected facts appear in the answer?
- **citation precision**: do citations point at correct sources?
- **latency p50 / p95**: the typical and the slow-tail response time

Use it to compare settings, e.g. reranker on vs off, or two models, on the same questions.

## 8. Testing

The tests replace the LLM with `ExtractiveFakeLLM` (a rule-based stand-in that picks sentences
by keyword overlap) and the embedding model with a hashing embedder. That makes the whole graph
run in milliseconds without a GPU or downloads, so CI can check behaviour on every push:
routing, rewrite limits, web fallback, abstention, citation mapping, persistence and the API.
Answer *quality* is what the evaluation harness measures, with the real model.

## 9. Questions you may be asked

**Why not just make the context window bigger and paste everything in?**
Cost and latency grow with context length, small local models have short windows (4K here), and
models get worse at using information buried in the middle of a long prompt. Retrieval keeps the
prompt small and focused.

**Why hybrid search instead of only embeddings?**
Embeddings miss exact-match terms (model numbers, acronyms, names) and BM25 misses paraphrases.
Fusing them covers both; RRF makes the fusion parameter-free.

**What does the grader add?**
It filters out irrelevant chunks before generation, and its verdict drives the loop: rewrite,
search the web, or answer. Without it the system cannot tell good retrieval from bad.

**How do you stop hallucination?**
Answers are restricted to retrieved sources, citations are validated against the actual source
list, unsupported questions end in an explicit abstention, and the evaluation set includes
unanswerable questions to measure it.

**Why LangGraph?**
The agent is a loop with branches. LangGraph makes the state, the nodes and the routing explicit,
caps the number of steps, and can export the graph as a diagram. Each node is still plain Python
and can be tested on its own.

**What would you do next?**
Stream tokens to the UI, add conversation memory, add an LLM-judged faithfulness metric, cache
embeddings of repeated queries, and swap the exact FAISS index for an approximate one (HNSW or
IVF) once the corpus grows past a few hundred thousand chunks.
