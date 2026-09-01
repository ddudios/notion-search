import notion_search
import agent

query = "에이전틱 루프 종료 조건"
corpus, embeddings = agent.corpus, agent.embeddings
idf = notion_search.compute_idf(corpus)

query_vec = notion_search.model.encode([query], normalize_embeddings=True)[0]
embedding_scores = embeddings @ query_vec

query_words = set(query.replace("?", "").split())
keyword_scores = [
    sum(idf.get(w, 0) for w in query_words & set(doc["text"].split()))
    for doc in corpus
]

for i, doc in enumerate(corpus):
    if doc["title"] == "1. AI Agent 개념":
        print(f"[정답 후보] {doc['title']}: 임베딩={embedding_scores[i]:.3f}, 키워드(정규화 전)={keyword_scores[i]:.3f}")

top_wrong_indices = sorted(range(len(corpus)), key=lambda i: -embedding_scores[i])[:3]
for i in top_wrong_indices:
    print(f"[상위 오답] {corpus[i]['title']}: 임베딩={embedding_scores[i]:.3f}, 키워드(정규화 전)={keyword_scores[i]:.3f}")

for i in top_wrong_indices:
    print(f"[상위 오답 내용] {corpus[i]['title']} ({len(corpus[i]['text'])}자): {corpus[i]['text']!r}")