import os
from haystack.document_stores import FAISSDocumentStore, ElasticsearchDocumentStore
import json
from haystack.schema import Document
from haystack.nodes import DensePassageRetriever, BM25Retriever, JoinAnswers, PreProcessor
from haystack import Pipeline

from rag_pipeline.document_stores import create_vector_store, create_es_store


DATA_FOLDER = "./data/"
INDEX_PATH = "./faiss_index"
FAISS_FILE = os.path.join(INDEX_PATH, "faiss_index")



def load_documents(data_folder=DATA_FOLDER):
    """Load all text/JSON/Markdown/Python files as Haystack Documents."""
    docs = []
    for fname in os.listdir(data_folder):
        if fname.endswith((".txt", ".md", ".py", ".json")):
            with open(os.path.join(data_folder, fname), "r", encoding="utf-8") as f:
                content = f.read()
                doc_id = os.path.splitext(fname)[0]
                docs.append(Document(content=content, meta={"filename": fname, "doc_id": doc_id}))
    return docs


def write_and_embed_documents(vector_store, documents, es_store):
    """Write only new documents to store and create embeddings for them."""

    FAISS_FILE = os.path.join(INDEX_PATH, "faiss_index")
    existing_filenames = {doc.meta["filename"] for doc in vector_store.get_all_documents()}
    new_docs = [doc for doc in documents if doc.meta["filename"] not in existing_filenames]

    if not new_docs:
        print("No new documents to add. Skipping write and embedding.")
        return None

    vector_store.write_documents(new_docs)
    print(f"Added {len(new_docs)} new documents to FAISS")

    es_store.write_documents(new_docs)
    print(f"Added {len(new_docs)} new documents to estore")

    retriever_dense = DensePassageRetriever(
        document_store=vector_store,
        query_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        passage_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    vector_store.update_embeddings(retriever_dense, update_existing_embeddings=False)
    vector_store.save(FAISS_FILE)
    print("Embeddings updated for new documents.")
    return retriever_dense

def load_nq_docs():
    DATA_FOLDER = "./data"
    os.makedirs(DATA_FOLDER, exist_ok=True)
    BEIR_NQ_FOLDER = "./datasets/nq"

    corpus_file = os.path.join(BEIR_NQ_FOLDER, "corpus.jsonl")
    documents = []

    with open(corpus_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc_json = json.loads(line)
            doc_id = doc_json["_id"]
            text = doc_json.get("text", "").strip()
            content = text
            # Prepare Haystack Document
            documents.append(Document(content=content, meta={"filename": f"{doc_id}.txt", "doc_id": doc_id}))


    print(f"Prepared {len(documents)} documents for indexing.")
    return documents


if __name__ == "__main__":
    #index the nq sample and generally nq
    documents = load_nq_docs()
    store = create_vector_store()
    es_store = create_es_store()
    write_and_embed_documents(store, documents, es_store)
    print("Indexing complete.")
