import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import notion_search
import agent
from eval_search import TEST_CASES

def get_rewritten_query(question: str) -> str:
    response = agent.client.messages.create(
        model=agent.MODEL,
        max_tokens=256,
        tools=agent.TOOLS,
        tool_choice={"type": "tool", "name": "search_notion"},
        messages=[{"role": "user", "content": question}],
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return tool_use.input["query"]

def run_eval(top_k: int = 3) -> None:
    corpus, embeddings = agent.corpus, agent.embeddings
    correct = 0

    for case in TEST_CASES:
        rewritten = get_rewritten_query(case["question"])
        results = notion_search.search(rewritten, corpus, embeddings, top_k=top_k)
        found_titles = [r["title"] for r in results]
        hit = case["expected_title"] in found_titles
        correct += hit
        status = "O" if hit else "X"
        print(f"[{status}] 원본: {case['question']!r} -> 재작성: {rewritten!r} -> {found_titles}")

    print(f"\nRecall@{top_k} (에이전트 경유): {correct}/{len(TEST_CASES)} ({correct / len(TEST_CASES):.0%})")

if __name__ == "__main__":
    run_eval()