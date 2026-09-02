import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import notion_search

TEST_CASES = [
    {"question": "에이전틱 루프의 종료 조건이 뭐야", "expected_title": "1. AI Agent 개념"},
    {"question": "임베딩이 뭐야", "expected_title": "4. RAG (검색 증강 생성)"},
    {"question": "하네스가 하는 일이 뭐야", "expected_title": "2. 하네스(Harness) 개념"},
]

def run_eval(top_k: int = 3) -> None:
    corpus, embeddings = notion_search.load_or_build()
    idf = notion_search.compute_idf(corpus)
    correct = 0

    for case in TEST_CASES:
        results = notion_search.search(case["question"], corpus, embeddings, idf, top_k=top_k)
        found_titles = [r["title"] for r in results]
        hit = case["expected_title"] in found_titles
        correct += hit
        status = "O" if hit else "X"
        print(f"[{status}] {case['question']} -> {found_titles}")

    print(f"\nRecall@{top_k}: {correct}/{len(TEST_CASES)} ({correct / len(TEST_CASES):.0%})")

if __name__ == "__main__":
    run_eval()