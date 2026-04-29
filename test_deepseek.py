#!/usr/bin/env python3
"""
测试DeepSeekLLM的基本功能
"""
import asyncio
from dotenv import load_dotenv
from deepseek_llm import DeepSeekLLM

async def test_deepseek_llm():
    load_dotenv()

    # 创建LLM实例
    llm = DeepSeekLLM(enable_reasoning=True)
    print("✅ DeepSeekLLM initialized successfully")
    print(f"Model: {llm.model}")
    print(f"Enable reasoning: {llm.enable_reasoning}")

    # 测试astream_invoke方法（不调用真实API）
    print("🔄 Testing astream_invoke method...")
    try:
        # 模拟调用，不使用真实消息
        messages = [{"role": "user", "content": "test"}]
        chunks = []
        async for chunk in llm.astream_invoke(messages):
            chunks.append(chunk)
            print("✅ astream_invoke method works!")
            print(f"Chunk received: {chunk[:100]}...")
            break  # 只测试第一个chunk

        if not chunks:
            print("❌ No chunks received from astream_invoke")

    except Exception as e:
        print(f"❌ astream_invoke method failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_deepseek_llm())