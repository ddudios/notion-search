import numpy as np
import notion_search
from eval_search import TEST_CASES

AI_CONCEPT_TITLES = {
    "1. AI Agent 개념",
    "2. 하네스(Harness) 개념",
    "3. 에이전트 오케스트레이션",
    "4. RAG (검색 증강 생성)",
    "5. 그 외 AI 기술",
}

def run_eval(top_k: int = 3) -> None:
    corpus, embeddings = notion_search.load_or_build()

    keep_indices = [i for i, doc in enumerate(corpus) if doc["title"] in AI_CONCEPT_TITLES]
    scoped_corpus = [corpus[i] for i in keep_indices]
    scoped_embeddings = embeddings[keep_indices]
    print(f"전체 {len(corpus)}개 청크 중 {len(scoped_corpus)}개로 범위 좁힘")

    correct = 0
    for case in TEST_CASES:
        results = notion_search.search(case["question"], scoped_corpus, scoped_embeddings, top_k=top_k)
        found_titles = [r["title"] for r in results]
        hit = case["expected_title"] in found_titles
        correct += hit
        status = "O" if hit else "X"
        print(f"[{status}] {case['question']} -> {found_titles}")

    print(f"\nRecall@{top_k} (5페이지로 범위 제한): {correct}/{len(TEST_CASES)} ({correct / len(TEST_CASES):.0%})")

if __name__ == "__main__":
    run_eval()