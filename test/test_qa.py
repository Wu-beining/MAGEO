import asyncio

from agent import QAAgent
from config.base import ModelConfig
from model.litellm_model import LiteLLMModel


async def main():
    # Load model config and create LLM
    config = ModelConfig.load()
    model = LiteLLMModel(**config.get_model("gpt-5-mini"))

    # Create the QAAgent (uses prompt templates by default)
    agent = QAAgent(model=model)

    # Example user query and documents
    user_query = "苹果公司2024年有哪些新产品？"
    documents = """
[1] 苹果公司于2024年9月发布了iPhone 16系列，包括iPhone 16、iPhone 16 Plus、iPhone 16 Pro和iPhone 16 Pro Max四款机型。主要亮点包括A18芯片、更大的屏幕尺寸和全新的相机控制功能。

[2] 2024年5月，苹果推出了新款iPad Pro和iPad Air，首次采用M4芯片，支持更薄的设计和OLED屏幕技术。

[3] Apple Watch Series 10于2024年9月发布，采用更薄的表身设计，新增睡眠呼吸暂停检测功能。

[4] 2024年10月，苹果发布了新款MacBook Pro，搭载M4 Pro和M4 Max芯片，性能比前代提升显著。
"""

    # Run QA
    result = await agent.run(user_query, documents)
    print("=" * 60)
    print("Answer:", result)
    print("=" * 60)

    # Quick sanity check
    assert len(result) > 0, "Answer should not be empty"
    assert "[1]" in result or "[2]" in result or "[3]" in result or "[4]" in result, \
        "Answer should contain citations"
    print("Test passed!")


if __name__ == "__main__":
    asyncio.run(main())
