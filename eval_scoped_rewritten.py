import notion_search
import agent
from eval_search import TEST_CASES
from eval_scoped import AI_CONCEPT_TITLES
from eval_agent import get_rewritten_query

def run_eval(top_k: int = 3) -> None:
    corpus, embeddings = agent.corpus, agent.embeddings

    keep_indices = [i for i, doc in enumerate(corpus) if doc["title"] in AI_CONCEPT_TITLES]
    scoped_corpus = [corpus[i] for i in keep_indices]
    scoped_embeddings = embeddings[keep_indices]
    print(f"전체 {len(corpus)}개 청크 중 {len(scoped_corpus)}개로 범위 좁힘")

    correct = 0
    for case in TEST_CASES:
        rewritten = get_rewritten_query(case["question"])
        results = notion_search.search(rewritten, scoped_corpus, scoped_embeddings, top_k=top_k)
        found_titles = [r["title"] for r in results]
        hit = case["expected_title"] in found_titles
        correct += hit
        status = "O" if hit else "X"
        print(f"[{status}] 원본: {case['question']!r} -> 재작성: {rewritten!r} -> {found_titles}")

    print(f"\nRecall@{top_k} (범위 좁힘 + 쿼리 재작성): {correct}/{len(TEST_CASES)} ({correct / len(TEST_CASES):.0%})")

if __name__ == "__main__":
    run_eval()