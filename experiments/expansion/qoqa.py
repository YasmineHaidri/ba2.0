#Paper inspiration:  https://arxiv.org/html/2407.12325v1
import logging

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Tuple

from src.openai_api_call import send_to_llm
from src.retrieval import retrieve_docs_only_hybrid

_QOQA_PROMPT_TEMPLATE = """
The user asked a question and we retrieved relevant paragraphs from our document collection in our knowledge base.
Below is the original question and the top passages related to it.

Use this information to generate rephrased and search queries that would better retrieve these passages.
The multiple search queries should be separated by a comma. Do not include any other commas except to separate the queries
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not use the words "definition", "meaning" unless present in the question.

Original question:
{question}

Sources:
{sources}

Search queries:

"""
QOQA_ITERATION_PROMPT_TEMPLATE ="""
The user asked a question and we retrieved relevant paragraphs from our document collection in our knowledge base.
Below is the original question, the top previously rephrased queries, and the most relevant passages related to the question.

Use this information to generate new rephrased and optimized search queries that would better retrieve these passages.
The multiple search queries should be separated by a comma. Do not include any other commas except to separate the queries.
Do not include cited source filenames and document names (e.g., info.txt or doc.pdf) in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not use the words "definition" or "meaning" unless they appear in the question.

Original question:
{question}

Top rephrased queries so far:
1. {top_query_1}
2. {top_query_2}
3. {top_query_3}

Sources:
{sources}

Search queries:
"""




def compute_bm25_scores(self, query: str, chunks: list[str]) -> list[float]:
        #https://github.com/dorianbrown/rank_bm25
        if not chunks:
            return []

        tokenized_chunks = [chunk.split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        tokenized_query = query.split()
        return list(bm25.get_scores(tokenized_query))


def compute_dense_scores(self, query: str, chunks: list[str]) -> list[float]:
        #https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
        model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        query_embedding = model.encode(query, convert_to_tensor=True)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
        return cosine_scores.cpu().tolist()


def compute_hybrid_alignment_score(self, query: str, chunks: list[str], alpha: float = 0.1) -> float:
        bm25_scores = self.compute_bm25_scores(query, chunks)
        dense_scores = self.compute_dense_scores(query, chunks)
        hybrid_scores = [alpha * b + d for b, d in zip(bm25_scores, dense_scores)]
        return float(np.mean(hybrid_scores))

async def get_optimized_query_with_qoqa(pipe,question: str):



        chunks=retrieve_docs_only_hybrid(pipe, question)
        max_iterations =3
        query_bucket = []
        _QOQA_PROMPT_TEMPLATE.replace("{sources}", chunks)
        QOQA_ITERATION_PROMPT_TEMPLATE.replace("{sources}", chunks)
        rephrased_queries = send_to_llm(
            user_prompt=_QOQA_PROMPT_TEMPLATE,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=80
        )
        for i in range(max_iterations):
            print(f'rephrased queries: {rephrased_queries}')
            rephrased_queries = [query.strip() for query in rephrased_queries.split(",") if query.strip()]
            print(f'rephrased queries: {rephrased_queries}')
            new_scored_queries = [
                (query, compute_hybrid_alignment_score(query, chunks, alpha= 0.1))
                for query in rephrased_queries
                ]
            query_bucket.extend(new_scored_queries)
            K=3
            top_queries = sorted(query_bucket, key=lambda x: x[1], reverse=True)[:K]
            QOQA_ITERATION_PROMPT_TEMPLATE.replace("{top_query_1}", top_queries[0][0])
            QOQA_ITERATION_PROMPT_TEMPLATE.replace("{top_query_2}", top_queries[1][0])
            QOQA_ITERATION_PROMPT_TEMPLATE.replace("{top_query_3}", top_queries[2][0])
            rephrased_queries = send_to_llm(
                    user_prompt=QOQA_ITERATION_PROMPT_TEMPLATE,
                    model="gpt-4o-mini",
                    temperature=0.3,
                    max_tokens=80
            )

        best_query, best_score = max(query_bucket, key=lambda x: x[1])
        print(f'best query is: {best_query}')
        return best_query


if __name__ == "__main__":
    store = load_document_store()

    es_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME, )

    pipe, bm25, dense, joiner = create_pipeline(store, es_store)
    query=get_optimized_query_with_qoqa(pipe, question='what is non controlling interest on balance sheet')
    print(query)


if __name__ == "__main__":
    store = load_document_store()

    es_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME, )

    pipe, bm25, dense, joiner = create_pipeline(store, es_store)
    query=get_optimized_query_with_qoqa(pipe, question='what is non controlling interest on balance sheet')
    print(query)


if __name__ == "__main__":
    store = load_document_store()

    es_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME, )

    pipe, bm25, dense, joiner = create_pipeline(store, es_store)
    query=get_optimized_query_with_qoqa(pipe, question='what is non controlling interest on balance sheet')
    print(query)


if __name__ == "__main__":
    store = load_document_store()

    es_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME, )

    pipe, bm25, dense, joiner = create_pipeline(store, es_store)
    query=get_optimized_query_with_qoqa(pipe, question='what is non controlling interest on balance sheet')
    print(query)