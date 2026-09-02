import json
import math
import os
import sys
from collections import Counter

from dotenv import load_dotenv
from notion_client import Client
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
notion = Client(auth=os.environ["NOTION_TOKEN"])
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def get_page_text(page_id: str) -> str:
    blocks = notion.blocks.children.list(block_id=page_id)["results"]
    texts = []
    for block in blocks:
        block_type = block["type"]
        rich_text = block.get(block_type, {}).get("rich_text", [])
        text = "".join([t["plain_text"] for t in rich_text])
        if text and text != "CONTENTS":
            texts.append(text)
    return "\n".join(texts)

HEADING_TYPES = {"heading_1", "heading_2", "heading_3"}

MIN_CHUNK_LENGTH = 40

def merge_short_chunks(chunks: list[str]) -> list[str]:
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHUNK_LENGTH:
            merged[-1] = merged[-1] + "\n" + chunk
        else:
            merged.append(chunk)
    return merged

def get_page_chunks(page_id: str) -> list[str]:
    blocks = notion.blocks.children.list(block_id=page_id)["results"]
    chunks = []
    current_lines = []
    started = False

    def flush():
        if current_lines:
            chunks.append("\n".join(current_lines))

    for block in blocks:
        block_type = block["type"]
        rich_text = block.get(block_type, {}).get("rich_text", [])
        text = "".join([t["plain_text"] for t in rich_text])
        if not text or text == "CONTENTS":
            continue
        if block_type in HEADING_TYPES:
            flush()
            current_lines = [text]
            started = True
        elif started:
            current_lines.append(text)
    flush()

    chunks = merge_short_chunks(chunks)

    if not chunks:
        full_text = get_page_text(page_id)
        if full_text:
            chunks = [full_text]

    return chunks

def get_page_title(page: dict) -> str:
    for prop in page.get("properties", {}).values():
        if prop["type"] == "title":
            texts = prop["title"]
            title = "".join([t["plain_text"] for t in texts])
            return title or "(제목 없음)"
    return "(제목 없음)"

def get_all_page_ids() -> list[dict]:
    pages = []
    cursor = None
    while True:
        response = notion.search(
            filter={"property": "object", "value": "page"},
            start_cursor=cursor,
        )
        for page in response["results"]:
            pages.append({
                "id": page["id"],
                "title": get_page_title(page),
                "last_edited_time": page["last_edited_time"],
            })
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return pages

def get_child_page_ids(page_id: str) -> list[dict]:
    blocks = notion.blocks.children.list(block_id=page_id)["results"]
    children = []
    for block in blocks:
        if block["type"] == "child_page":
            children.append({
                "id": block["id"],
                "title": block["child_page"]["title"],
            })
    return children

def build_corpus() -> list[dict]:
    corpus = []
    pages = get_all_page_ids()
    for page in pages:
        chunks = get_page_chunks(page["id"])
        for chunk_text in chunks:
            corpus.append({
                "title": page["title"],
                "id": page["id"],
                "text": chunk_text,
            })
    return corpus

CACHE_DIR = "cache"
CORPUS_CACHE_PATH = os.path.join(CACHE_DIR, "corpus.json")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
PAGE_META_PATH = os.path.join(CACHE_DIR, "page_meta.json")

def load_or_build():
    old_corpus, old_embeddings, old_meta = [], None, {}

    if os.path.exists(CORPUS_CACHE_PATH) and os.path.exists(EMBEDDINGS_CACHE_PATH):
        with open(CORPUS_CACHE_PATH, "r", encoding="utf-8") as f:
            old_corpus = json.load(f)
        old_embeddings = np.load(EMBEDDINGS_CACHE_PATH)
        if os.path.exists(PAGE_META_PATH):
            with open(PAGE_META_PATH, "r", encoding="utf-8") as f:
                old_meta = json.load(f)

    current_pages = get_all_page_ids()
    current_ids = {p["id"] for p in current_pages}
    changed_pages = [
        p for p in current_pages
        if old_meta.get(p["id"]) != p["last_edited_time"]
    ]
    print(f"전체 {len(current_pages)}개 페이지 중 {len(changed_pages)}개 변경/신규 감지")

    changed_ids = {p["id"] for p in changed_pages}
    keep_mask = [doc["id"] in current_ids and doc["id"] not in changed_ids for doc in old_corpus]
    kept_corpus = [doc for doc, keep in zip(old_corpus, keep_mask) if keep]
    kept_embeddings = old_embeddings[keep_mask] if old_embeddings is not None and len(old_corpus) else None

    new_corpus = []
    for page in changed_pages:
        for chunk_text in get_page_chunks(page["id"]):
            new_corpus.append({"title": page["title"], "id": page["id"], "text": chunk_text})
    new_embeddings = embed_corpus(new_corpus) if new_corpus else None

    corpus = kept_corpus + new_corpus
    if kept_embeddings is not None and new_embeddings is not None:
        embeddings = np.vstack([kept_embeddings, new_embeddings])
    elif new_embeddings is not None:
        embeddings = new_embeddings
    else:
        embeddings = kept_embeddings

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CORPUS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    with open(PAGE_META_PATH, "w", encoding="utf-8") as f:
        json.dump({p["id"]: p["last_edited_time"] for p in current_pages}, f, ensure_ascii=False)

    return corpus, embeddings

def embed_corpus(corpus: list[dict]) -> np.ndarray:
      texts = [doc["text"] for doc in corpus]
      return model.encode(texts, normalize_embeddings=True)

def compute_idf(corpus: list[dict]) -> dict:
    doc_count = len(corpus)
    doc_freq = Counter()
    for doc in corpus:
        for word in set(doc["text"].split()):
            doc_freq[word] += 1
    return {word: math.log(doc_count / freq) for word, freq in doc_freq.items()}

def search(query: str, corpus: list[dict], embeddings: np.ndarray, idf: dict, top_k: int = 3) -> list[dict]:
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    embedding_scores = embeddings @ query_vec

    length_factor = np.array([min(1.0, len(doc["text"]) / 30) for doc in corpus])
    embedding_scores = embedding_scores * length_factor

    query_words = set(query.replace("?", "").split())
    keyword_scores = np.array([
        sum(idf.get(w, 0) for w in query_words & set(doc["text"].split()))
        for doc in corpus
    ])
    if keyword_scores.max() > 0:
        keyword_scores = keyword_scores / keyword_scores.max()

    final_scores = embedding_scores + 0.3 * keyword_scores
    top_indices = np.argsort(-final_scores)[:top_k]
    return [{**corpus[i], "score": float(final_scores[i])} for i in top_indices]

if __name__ == "__main__":
    corpus, embeddings = load_or_build()
    idf = compute_idf(corpus)
    print(f"총 {len(corpus)}개 청크")
    print("완료! 질문을 입력하세요 (종료: exit)")
    while True:
        query = input("> ")
        if query == "exit":
            break
        if not query.strip():
            continue
        results = search(query, corpus, embeddings, idf, top_k=3)
        for r in results:
            print(f"\n[{r['title']}] (유사도 {r['score']:.3f})")
            print(r["text"][:200])