import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

try:
    from zai import ZhipuAiClient
except ModuleNotFoundError:
    ZhipuAiClient = None

load_dotenv()


def web_search(query):
    if ZhipuAiClient is None:
        raise ModuleNotFoundError(
            "web_search requires the optional dependency `zai-sdk`. "
            "Install it before running the interactive search pipeline."
        )

    api_key = os.getenv("WEB_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("WEB_SEARCH_API_KEY is not set.")

    client = ZhipuAiClient(api_key=api_key)
    response = client.web_search.web_search(
        search_engine="search_pro",
        search_query=query,
        count=10,
    )
    search_results = response.search_result

    # 整理search_results为字典 [{"index": 0, "title":"", "content":"", "link":""}]
    formatted_results = []
    for i, result in enumerate(search_results):
        formatted_results.append(
            {
                "index": i,
                "title": result.title,
                "content": result.content,
                "link": result.link,
            }
        )

    # 保存结果到本地 log/web_search 文件夹
    log_dir = Path("log/web_search")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = log_dir / f"{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            {"query": query, "results": formatted_results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return formatted_results
