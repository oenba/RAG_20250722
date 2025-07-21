from es_client import es
now_ts = int(datetime.utcnow().timestamp() * 1000)
from datetime import datetime

INDEX_NAME = "rag_documents"

def search_with_hybrid_score(query: str, embedding: list[float], now_ts: int, top_k: int = 10):
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "script": {
                    "source": """
                        double bm25 = _score;
                        double norm_bm25 = Math.log(1 + bm25) / Math.log(11);

                        double cosine = cosineSimilarity(params.query_vector, 'embedding_vector');
                        double norm_cosine = (cosine + 1.0) / 2.0;

                        double hours = (params.now - doc['publish_time'].value.toInstant().toEpochMilli()) / 3600000.0;
                        double time_boost = 1.0 / (1.0 + hours / 24.0);

                        return 0.5 * norm_bm25 + 0.4 * norm_cosine + 0.1 * time_boost;
                    """,
                    "params": {
                        "query_vector": embedding,
                        "now": now_ts
                    }
                }
            }
        }
    }

    response = es.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]
    return [
        {
            "id": hit["_id"],
            "score": hit["_score"],
            "content": hit["_source"].get("content", ""),
            "publish_time": hit["_source"].get("publish_time", "")
        }
        for hit in hits
    ]




场景	α (BM25)	β (Cosine)	γ (Extra)
FAQ型问答（命中关键词为主）	0.7	0.3	0.0
语义问答（开放性）	0.4	0.6	0.0
新闻类（关注时间）	0.5	0.4	0.1
复杂问答（需 LTR）	LTR 替代上式，或加入特征分





Enterprise-Grade Hybrid Scoring Strategy for RAG

This strategy is designed to combine keyword relevance (BM25), semantic similarity (cosine similarity), and optional business-specific features into a unified ranking score for use in Retrieval-Augmented Generation (RAG) systems.


