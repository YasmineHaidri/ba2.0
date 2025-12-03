import os
from haystack.document_stores import FAISSDocumentStore, ElasticsearchDocumentStore
from haystack.schema import Document


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
FAISS_FILE = os.path.join(INDEX_PATH, "faiss_index")

ES_INDEX_NAME = "bm25_docs"
ES_HOST = "localhost"
ES_PORT = 9200





def create_vector_store():
    """Create or load a persistent FAISS document store. It will be used for dense retrieval"""
    vector_db = FAISSDocumentStore(
        faiss_index_factory_str="Flat",
        sql_url=f"sqlite:///{INDEX_PATH}/faiss_docs.db",
        return_embedding=True,
        embedding_dim=384
    )
    return vector_db


def load_vector_store():
    """Load existing FAISS index from disk."""
    return FAISSDocumentStore.load(FAISS_FILE)


def create_es_store():
    '''
    Elastic Search store is for keyword search or BM25 no embedding is happening here
    :return:
    '''
    return ElasticsearchDocumentStore(
        host="localhost",
        index=ES_INDEX_NAME
    )

def load_es_store():
    es_store = (ElasticsearchDocumentStore
                (host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME,
                                          ))
    return es_store