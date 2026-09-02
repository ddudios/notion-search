import os
from dotenv import load_dotenv
import anthropic
import notion_search

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5"

print("코퍼스 준비 중...")
corpus, embeddings = notion_search.load_or_build()
idf = notion_search.compute_idf(corpus)
print(f"준비 완료 ({len(corpus)}개 청크)")

TOOLS = [
    {
        "name": "search_notion",
        "description": "사용자의 노션 메모에서 질문과 관련된 내용을 검색한다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 내용을 짧은 키워드가 아니라, 구체적이고 완전한 문장으로 작성하라"}
            },
            "required": ["query"],
        },
    }
]

def execute_tool(name: str, tool_input: dict) -> str:
    if name == "search_notion":
        results = notion_search.search(tool_input["query"], corpus, embeddings, idf, top_k=3)
        return "\n\n---\n\n".join([f"[{r['title']}]\n{r['text']}" for r in results])
    return f"알 수 없는 도구: {name}"

def run(user_message: str) -> None:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = next((b.text for b in response.content if b.type == "text"), "")
            print(f"\n===== 최종 답변 =====")
            print(final_text)
            return final_text

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        tool_results = []
        for block in tool_use_blocks:
            print(f"[요청] {block.name}({block.input})")
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    question = input("질문: ")
    run(question)