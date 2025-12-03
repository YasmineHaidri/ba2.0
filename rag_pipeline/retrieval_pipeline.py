import os

from haystack.document_stores import FAISSDocumentStore, ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, DensePassageRetriever, JoinDocuments
from haystack import Pipeline



def create_pipeline(store, es_store, dense_model="sentence-transformers/all-MiniLM-L6-v2"):
    bm25 = BM25Retriever(document_store=es_store)
    dense = DensePassageRetriever(
        document_store=store,
        query_embedding_model=dense_model,
        passage_embedding_model=dense_model,
        use_gpu=True,
    )

    joiner = JoinDocuments(join_mode="merge")
    pipe = Pipeline()
    pipe.add_node(component=bm25, name="bm25", inputs=["Query"])
    pipe.add_node(component= dense, name="dpr", inputs=["Query"])
    pipe.add_node(component=joiner, name="joiner", inputs=["bm25", "dpr"])
    return pipe

def retrieve_docs(pipe, query, top_k=10):
    """
    Run the pipeline and return per-retriever and hybrid retrieval results.
    Returns:
        {
            "bm25": [...],
            "dense": [...],
            "hybrid": [...]
        }
    """
    out = pipe.run(
        query=query,
        params={
            "bm25": {"top_k": top_k},
            "dpr": {"top_k": top_k}
        },
        debug=True
    )

    hybrid_docs = out.get("documents", [])[:top_k]

    debug_data = out.get("_debug", {})

    bm25_docs = []
    dpr_docs = []

    if "bm25" in debug_data:
        bm25_docs = debug_data["bm25"].get("output", {}).get("documents", [])[:top_k]
    if "dpr" in debug_data:
        dpr_docs = debug_data["dpr"].get("output", {}).get("documents", [])[:top_k]

    def pick(docs):
        results = []
        for doc in docs:
            results.append({
                "doc_id": doc.meta.get("doc_id"),
                "score": float(doc.score) if doc.score is not None else 0.0,
            })
        return results

    return {
        "bm25": pick(bm25_docs),
        "dense": pick(dpr_docs),
        "hybrid": pick(hybrid_docs)
    }

def retrieve_docs_only_hybrid(pipe, query, top_k=10):
    """Run the hybrid pipeline and return doc_id, score, and content."""
    out = pipe.run(
        query=query,
        params={"bm25": {"top_k": top_k},
                "dpr": {"top_k": top_k}}
    )
    docs = out["documents"][:top_k]
    return [doc.content for doc in docs]

if __name__ == "__main__":
    store = load_document_store()

    #es_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME, )

   # pipe, bm25, dense, joiner = create_pipeline(store, es_store)
   # test_results=retrieve_docs(pipe, 'who sings love will keep us alive by the eagles')
   # with open("results.json", "w", encoding="utf-8") as f:
      #  json.dump(test_results, f, indent=2, ensure_ascii=False)
