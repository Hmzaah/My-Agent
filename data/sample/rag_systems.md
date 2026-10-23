# Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation (RAG) answers a question by first retrieving relevant passages from a document collection and then asking a language model to answer using those passages. Grounding the model in retrieved text reduces hallucination and lets the system answer questions about private or recent documents the model was never trained on.

## Indexing

Documents are split into chunks, typically a few hundred tokens long, often with some overlap so that facts crossing a boundary are not lost. Each chunk is converted into a dense vector by an embedding model, and the vectors are stored in a vector index such as FAISS for fast nearest-neighbour search.

## Retrieval

Dense retrieval finds chunks whose embeddings are close to the query embedding, which captures meaning and paraphrases. Sparse retrieval with BM25 scores chunks by exact term overlap, which works better for names, codes and rare technical terms. Hybrid retrieval runs both and merges the results. Reciprocal rank fusion (RRF) is a simple merging method: each document receives the sum of 1 / (k + rank) over the ranked lists it appears in, with k commonly set to 60.

## Reranking

A cross-encoder reranker reads the query and a candidate passage together and outputs a relevance score. It is more accurate than comparing separate embeddings but slower, so it is applied only to the top candidates from the first retrieval stage.

## Agentic RAG

Agentic RAG adds decision steps around retrieval: grading whether the retrieved passages actually answer the question, rewriting the query and searching again when they do not, falling back to other tools such as web search, and refusing to answer when no reliable evidence is found.

## Evaluation

Retrieval is commonly evaluated with hit rate at k (whether a relevant passage appears in the top k results) and mean reciprocal rank (MRR). Answers are evaluated for correctness, faithfulness to the retrieved sources, and citation accuracy.
