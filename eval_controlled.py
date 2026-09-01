import notion_search
import agent
from eval_scoped import AI_CONCEPT_TITLES

# eval_scoped_rewritten.py가 만들어낸 재작성 쿼리를 그대로 고정값으로 사용
FIXED_CASES = [
    {"query": "에이전틱 루프 종료 조건", "expected_title": "1. AI Agent 개념"},
    {"query": "임베딩이란 무엇인가 embedding의 정의와 개념", "expected_title": "4. RAG (검색 증강 생성)"},
    {"query": "하네스가 하는 일이 뭐야 하네스의 역할과 업무", "expected_title": "2. 하네스(Harness) 개념"},
]

def check(corpus, embeddings, idf, label: str, top_k: int = 3) -> None:
    correct = 0
    for case in FIXED_CASES:
        results = notion_search.search(case["query"], corpus, embeddings, idf, top_k=top_k)
        found_titles = [r["title"] for r in results]
        hit = case["expected_title"] in found_titles
        correct += hit
        status = "O" if hit else "X"
        print(f"[{status}] {case['query']!r} -> {found_titles}")
    print(f"Recall@{top_k} ({label}): {correct}/{len(FIXED_CASES)} ({correct / len(FIXED_CASES):.0%})\n")

def run() -> None:
    full_corpus, full_embeddings = agent.corpus, agent.embeddings
    full_idf = notion_search.compute_idf(full_corpus)

    keep_indices = [i for i, doc in enumerate(full_corpus) if doc["title"] in AI_CONCEPT_TITLES]
    scoped_corpus = [full_corpus[i] for i in keep_indices]
    scoped_embeddings = full_embeddings[keep_indices]
    scoped_idf = notion_search.compute_idf(scoped_corpus)

    print("=== 전체 코퍼스 (하이브리드 검색 적용) ===")
    check(full_corpus, full_embeddings, full_idf, "전체 코퍼스")

    print("=== 5페이지로 좁힌 코퍼스 ===")
    check(scoped_corpus, scoped_embeddings, scoped_idf, "좁힌 코퍼스")

if __name__ == "__main__":
    run()