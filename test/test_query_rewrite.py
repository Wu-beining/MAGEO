import asyncio
import json

from dotenv import load_dotenv

from agent import QueryRewriteAgent
from config.base import ModelConfig
from model.litellm_model import LiteLLMModel

load_dotenv()


async def main():
    # Load model config and create LLM
    config = ModelConfig.load()
    model = LiteLLMModel(**config.get_model("gemini-2.5-pro"))

    # Create the QueryRewriteAgent (uses prompt templates by default)
    agent = QueryRewriteAgent(model=model, ensure_json=True)

    # Example user query
    user_query = "帮我查下2024年苹果新品发布会的时间和亮点，谢谢"

    # Run rewrite
    result = await agent.run(user_query)
    print("Rewrite (JSON string):", result)

    # Optional: parse as dict and do a quick sanity check
    data = json.loads(result)
    assert "main_query" in data and isinstance(data["main_query"], str)
    assert "alternative_queries" in data and isinstance(
        data["alternative_queries"], list
    )
    print("Parsed:", data)


if __name__ == "__main__":
    asyncio.run(main())
