"""✏️ ส่วนที่ 3: Retrieval — ค้นหา chunk ที่เกี่ยวข้องกับคำถาม

ค่าตั้งต้นตอนยังไม่ทำ: ไม่คืน chunk ใดเลย LLM จึงไม่เห็นเอกสาร
"""
import re
from core.types import Chunk
from core.utils import cosine_many
from student import settings

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def _tokenize(text: str) -> list[str]:
    """แยกคำภาษาไทยและภาษาอังกฤษสำหรับ BM25 และ keyword search"""
    text_lower = text.lower()
    tokens = re.findall(r"[a-z0-9_-]+", text_lower)

    thai_runs = re.findall(r"[\u0e00-\u0e7f]+", text_lower)
    for run in thai_runs:
        tokens.append(run)
        if len(run) > 3:
            for i in range(len(run) - 1):
                tokens.append(run[i:i + 2])
            for i in range(len(run) - 2):
                tokens.append(run[i:i + 3])
    return tokens if tokens else [text_lower]


def make_query(question: str, options: list[str]) -> str:
    """สร้างข้อความที่จะใช้ค้นหา"""
    q = question.strip()
    if getattr(settings, "QUERY_WITH_OPTIONS", True) and options:
        opt_str = " ".join(o.strip() for o in options if o.strip())
        if opt_str:
            q = f"{q} {opt_str}"
    return q if q else "query"


def search(chunks: list[Chunk], query_text: str, query_vector: list[float] | None, k: int) -> list[tuple[Chunk, float]]:
    """เลือก chunk ที่เกี่ยวข้องที่สุด k ชิ้น ด้วย Hybrid Search (Vector + BM25)"""
    if not chunks:
        return []

    target_k = getattr(settings, "TOP_K", k) or k
    hybrid_weight = getattr(settings, "HYBRID_WEIGHT", 0.4)
    min_score = getattr(settings, "MIN_SCORE", None)

    # 1. Vector cosine similarity
    vector_scores = [0.0] * len(chunks)
    has_vectors = query_vector is not None and any(c.vector is not None for c in chunks)
    if has_vectors:
        valid_indices = [i for i, c in enumerate(chunks) if c.vector is not None]
        if valid_indices:
            sub_vectors = [chunks[i].vector for i in valid_indices]
            sub_scores = cosine_many(query_vector, sub_vectors)
            for idx, sc in zip(valid_indices, sub_scores):
                vector_scores[idx] = max(0.0, (sc + 1.0) / 2.0)

    # 2. BM25 / Keyword scoring
    keyword_scores = [0.0] * len(chunks)
    query_tokens = _tokenize(query_text)
    if BM25Okapi is not None and query_tokens:
        corpus = [_tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(corpus)
        raw_bm25 = bm25.get_scores(query_tokens)
        max_bm25 = max(raw_bm25) if len(raw_bm25) > 0 else 0.0
        if max_bm25 > 0:
            keyword_scores = [float(s / max_bm25) for s in raw_bm25]
    elif query_tokens:
        q_set = set(query_tokens)
        raw_kw = []
        for c in chunks:
            c_tokens = _tokenize(c.text)
            overlap = sum(1 for t in c_tokens if t in q_set)
            raw_kw.append(overlap)
        max_kw = max(raw_kw) if raw_kw else 0
        if max_kw > 0:
            keyword_scores = [float(s / max_kw) for s in raw_kw]

    # 3. รวมคะแนน
    combined: list[tuple[Chunk, float]] = []
    for i, c in enumerate(chunks):
        if has_vectors and hybrid_weight > 0:
            final_score = (1.0 - hybrid_weight) * vector_scores[i] + hybrid_weight * keyword_scores[i]
        elif has_vectors:
            final_score = vector_scores[i]
        else:
            final_score = keyword_scores[i]

        if min_score is not None and final_score < min_score:
            continue
        combined.append((c, float(final_score)))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:target_k]
