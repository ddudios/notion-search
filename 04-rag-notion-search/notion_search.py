import sys
import json
import os
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
            pages.append({"id": page["id"], "title": get_page_title(page)})
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

def load_or_build():
    if os.path.exists(CORPUS_CACHE_PATH) and os.path.exists(EMBEDDINGS_CACHE_PATH):
        print("캐시에서 불러오는 중...")
        with open(CORPUS_CACHE_PATH, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        embeddings = np.load(EMBEDDINGS_CACHE_PATH)
        return corpus, embeddings

    print("캐시 없음, 새로 만드는 중...")
    corpus = build_corpus()
    embeddings = embed_corpus(corpus)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CORPUS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)

    return corpus, embeddings

def embed_corpus(corpus: list[dict]) -> np.ndarray:
      texts = [doc["text"] for doc in corpus]
      return model.encode(texts, normalize_embeddings=True)

def search(query: str, corpus: list[dict], embeddings: np.ndarray, top_k: int = 3) -> list[dict]:
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec
    top_indices = np.argsort(-scores)[:top_k]
    return [{**corpus[i], "score": float(scores[i])} for i in top_indices]

if __name__ == "__main__":
    corpus, embeddings = load_or_build()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = search(query, corpus, embeddings, top_k=3)
        for r in results:
            print(f"\n[{r['title']}] (유사도 {r['score']:.3f})")
            print(r["text"][:200])
    else:
        print(f"총 {len(corpus)}개 청크")
        print("완료! 질문을 입력하세요 (종료: exit)")
        while True:
            query = input("> ")
            if query == "exit":
                break
            if not query.strip():
                continue
            results = search(query, corpus, embeddings, top_k=3)
            for r in results:
                print(f"\n[{r['title']}] (유사도 {r['score']:.3f})")
                print(r["text"][:200])