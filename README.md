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


FinalScore = α * NormBM25 + β * NormCosine + γ * ExtraScore
α, β, γ are tunable weights.

✅ 1. Unified Scoring Formula
The final document relevance score is computed as a weighted combination:

FinalScore = α * NormBM25 + β * NormCosine + γ * ExtraScore
α, β, γ are tunable weights.

NormBM25 is the normalized BM25 relevance score.

NormCosine is the normalized cosine similarity score.

ExtraScore is an optional field (e.g., time-based decay, popularity score, user engagement).

✅ 2. Score Normalization Techniques
🔹 BM25 Normalization
BM25 scores (_score in Elasticsearch) do not have a fixed range, so normalization is required.

 Log-Based Normalization (recommended for ES)

norm_bm25 = Math.log(1 + _score) / Math.log(1 + bm25_max)
You can assume bm25_max = 10.0 as a starting point.

🔹 Cosine Similarity Normalization
Elasticsearch’s cosine similarity returns values between -1 and 1, so we shift and scale it:


double cosine = cosineSimilarity(params.query_vector, 'embedding_vector');
double norm_cosine = (cosine + 1.0) / 2.0;  // Result: [0, 1]

🔹 Extra Feature Score (Optional)
For example, time decay scoring:


double age_ms = params.now - doc['publish_time'].value.toInstant().toEpochMilli();
double hours = age_ms / 3600000.0;
double time_score = 1.0 / (1.0 + hours / 24.0);  // Prefer recent documents
✅ 3. Elasticsearch script_score Query (Full Example)

{
  "size": 10,
  "query": {
    "script_score": {
      "query": {
        "match": {
          "content": "{{ your_query }}"
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
          "query_vector": [0.1, 0.2, 0.3, ...],
          "now": 1690000000000
        }
      }
    }
  }
}
✅ 4. Recommended Weight Settings by Scenario
Use Case	α (BM25)	β (Cosine)	γ (Time/Extra)
FAQ-style retrieval	0.7	0.3	0.0
Open-ended semantic query	0.4	0.6	0.0
News/article ranking	0.5	0.4	0.1
Financial/legal QA	0.5	0.5	0.0–0.1

✅ 5. Optional Enhancements
Cross-Encoder Re-ranking: Use a BERT model to re-rank Top-N retrieved passages with full semantic matching.

Learning-to-Rank (LTR): Use XGBoost, LightGBM, or neural models trained on feedback (clicks, ratings).

Dynamic weight control: Tune α/β/γ based on query type or user profile.

Vector quantization or ANN indexing for faster retrieval at scale.




| Use Case                  | α (BM25) | β (Cosine) | γ (Time/Extra) |
| ------------------------- | -------- | ---------- | -------------- |
| FAQ-style retrieval       | 0.7      | 0.3        | 0.0            |
| Open-ended semantic query | 0.4      | 0.6        | 0.0            |
| News/article ranking      | 0.5      | 0.4        | 0.1            |
| Financial/legal QA        | 0.5      | 0.5        | 0.0–0.1        |





def split_text_with_metadata(text: str, source: str, page: int = None):
    chunks = splitter.split_text(text)
    return [
        {
            "content": chunk,
            "tokens": tiktoken_len(chunk),
            "metadata": {
                "source": source,
                "page": page
            }
        }
        for chunk in chunks
    ]




from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2TokenizerFast

# Tokenizer compatible with Google text-embedding-005
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def tiktoken_len(text: str) -> int:
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,             # target ~350 tokens
    chunk_overlap=50,           # allow 50 token overlap
    length_function=tiktoken_len,
    separators=["\n\n", "\n", ".", " ", ""],  # hierarchy of splitting
)

def split_text(text: str):
    return splitter.split_text(text)

