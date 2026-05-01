"""LLM适配器 - 支持OpenAI、Anthropic、Gemini等不同接口格式"""

import time
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Iterator, List, Dict, Any, Union, AsyncIterator

from .llm_response import LLMResponse, StreamStats
from .exceptions import HelloAgentsException


class BaseLLMAdapter(ABC):
    """LLM适配器基类"""

    def __init__(self, api_key: str, base_url: Optional[str], timeout: int, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        self._client = None
        self._async_client = None

    @abstractmethod
    def create_client(self) -> Any:
        """创建客户端实例"""
        pass

    def create_async_client(self) -> Any:
        """创建异步客户端实例（子类可选实现）"""
        return None

    @abstractmethod
    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        pass

    @abstractmethod
    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用，返回生成器"""
        pass

    async def astream_invoke(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        """异步流式调用（子类可选实现真正的异步）

        默认实现：使用队列 + 线程池包装同步流式方法
        """
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream_to_queue():
            try:
                for chunk in self.stream_invoke(messages, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # 在线程池中运行同步流式方法
        loop.run_in_executor(None, _stream_to_queue)

        # 从队列中逐个取出 chunk
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    @abstractmethod
    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """工具调用（Function Calling）"""
        pass

    async def astream_invoke_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: Union[str, Dict] = "auto",
        **kwargs
    ):
        """
        异步流式工具调用 —— 默认降级为非流式 ainvoke_with_tools。
        子类（OpenAI/DeepSeek）会重写为真正的流式版本。

        Yields:
            dict，固定格式：
              {"type": "text",       "content": str}          # LLM 文本 delta
              {"type": "thinking",   "content": str}          # 推理过程 delta（thinking model）
              {"type": "tool_calls", "tool_calls": list}      # 完整 tool_calls 列表（流结束后）
              {"type": "done",       "response": response}    # 原始响应对象（用于后续处理）
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.invoke_with_tools(messages, tools, tool_choice, **kwargs)
        )
        # 把非流式响应包装成统一格式
        msg = response.choices[0].message
        if msg.content:
            yield {"type": "text", "content": msg.content}
        if msg.tool_calls:
            yield {"type": "tool_calls", "tool_calls": msg.tool_calls}
        yield {"type": "done", "response": response}

    def _is_thinking_model(self, model_name: str) -> bool:
        """判断是否为thinking model"""
        thinking_keywords = ["reasoner", "o1", "o3", "thinking"]
        model_lower = model_name.lower()
        return any(keyword in model_lower for keyword in thinking_keywords)


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI兼容接口适配器（默认）

    支持：
    - OpenAI官方API
    - 所有OpenAI兼容接口（DeepSeek、Qwen、Kimi、智谱等）
    - Thinking Models（o1、deepseek-reasoner等）
    """

    def create_client(self) -> Any:
        """创建OpenAI客户端"""
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def create_async_client(self) -> Any:
        """创建OpenAI异步客户端"""
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()
        
        start_time = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # 提取内容和推理过程
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning_content = None
            
            # Thinking model特殊处理
            if self._is_thinking_model(self.model):
                # OpenAI o1系列：reasoning_content在message中
                if hasattr(choice.message, 'reasoning_content'):
                    reasoning_content = choice.message.reasoning_content
                # DeepSeek reasoner：可能在其他字段
                elif hasattr(choice, 'reasoning_content'):
                    reasoning_content = choice.reasoning_content
            
            # 提取usage信息
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )
            
        except Exception as e:
            raise HelloAgentsException(f"OpenAI API调用失败: {str(e)}")
    
    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()
        
        start_time = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            collected_content = []
            reasoning_content = None
            usage = {}
            
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    
                    # 提取内容
                    if delta.content:
                        collected_content.append(delta.content)
                        yield delta.content
                    
                    # Thinking model的推理过程
                    if self._is_thinking_model(self.model):
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            if reasoning_content is None:
                                reasoning_content = ""
                            reasoning_content += delta.reasoning_content

                # 提取usage（流式最后一个chunk可能包含）
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            # 返回统计信息（存储到适配器，供外部获取）
            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )

        except Exception as e:
            raise HelloAgentsException(f"OpenAI API流式调用失败: {str(e)}")

    async def astream_invoke(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        """真正的异步流式调用（使用 OpenAI 原生异步客户端）"""
        if not self._async_client:
            self._async_client = self.create_async_client()

        start_time = time.time()

        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )

            collected_content = []
            reasoning_content = None
            usage = {}

            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # 提取内容
                    if delta.content:
                        collected_content.append(delta.content)
                        yield delta.content

                    # Thinking model的推理过程
                    if self._is_thinking_model(self.model):
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            if reasoning_content is None:
                                reasoning_content = ""
                            reasoning_content += delta.reasoning_content

                # 提取usage（流式最后一个chunk可能包含）
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            # 返回统计信息（存储到适配器，供外部获取）
            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content
            )

        except Exception as e:
            raise HelloAgentsException(f"OpenAI API异步流式调用失败: {str(e)}")

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict],
                         tool_choice: Union[str, Dict] = "auto", **kwargs) -> Any:
        """工具调用（Function Calling）"""
        if not self._client:
            self._client = self.create_client()

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs
            )
            return response

        except Exception as e:
            raise HelloAgentsException(f"OpenAI Function Calling调用失败: {str(e)}")

    async def astream_invoke_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: Union[str, Dict] = "auto",
        **kwargs
    ):
        """
        OpenAI 真流式工具调用。

        同时支持：
        - 文本 delta 实时 yield（LLM 思考/回复）
        - tool_calls delta 收集并在流结束后 yield 完整列表
        - 兼容 reasoning_content（thinking model）

        Yields:
            {"type": "text",       "content": str}
            {"type": "thinking",   "content": str}
            {"type": "tool_calls", "tool_calls": list}
            {"type": "done",       "response": None}
        """
        if not self._async_client:
            self._async_client = self.create_async_client()

        try:
            stream = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
                **kwargs
            )

            # 收集 tool_calls delta（按 index 拼接）
            tool_calls_map: Dict[int, Dict] = {}

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # ── 文本 delta ──────────────────────────────────
                if delta.content:
                    yield {"type": "text", "content": delta.content}

                # ── 推理 delta（thinking model）────────────────
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    yield {"type": "thinking", "content": rc}

                # ── tool_calls delta ────────────────────────────
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        entry = tool_calls_map[idx]
                        if tc_delta.id:
                            entry["id"] += tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments

            # 流结束 —— 输出完整 tool_calls
            if tool_calls_map:
                tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)]
                yield {"type": "tool_calls", "tool_calls": tool_calls}

            yield {"type": "done", "response": None}

        except Exception as e:
            raise HelloAgentsException(f"OpenAI 流式 Function Calling 失败: {str(e)}")


class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude适配器

    处理Claude特有的消息格式：
    - system参数独立（不在messages中）
    - 消息格式转换
    """

    def create_client(self) -> Any:
        """创建Anthropic客户端"""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise HelloAgentsException(
                "使用Anthropic需要安装: pip install anthropic"
            )

        return Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _convert_messages(self, messages: List[Dict]) -> tuple[Optional[str], List[Dict]]:
        """转换消息格式，提取system消息"""
        system_content = None
        converted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                converted_messages.append(msg)

        return system_content, converted_messages

    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_content, converted_messages = self._convert_messages(messages)

        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs
            }
            if system_content:
                request_params["system"] = system_content

            response = self._client.messages.create(**request_params)

            latency_ms = int((time.time() - start_time) * 1000)

            # 提取内容
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text

            # 提取usage
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise HelloAgentsException(f"Anthropic API调用失败: {str(e)}")

    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_content, converted_messages = self._convert_messages(messages)

        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                "stream": True,
                **kwargs
            }
            if system_content:
                request_params["system"] = system_content

            usage = {}

            with self._client.messages.stream(**request_params) as stream:
                for text in stream.text_stream:
                    yield text

                # 获取最终消息以提取usage
                final_message = stream.get_final_message()
                if hasattr(final_message, 'usage') and final_message.usage:
                    usage = {
                        "prompt_tokens": final_message.usage.input_tokens,
                        "completion_tokens": final_message.usage.output_tokens,
                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise HelloAgentsException(f"Anthropic API流式调用失败: {str(e)}")

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """工具调用（Anthropic格式）"""
        if not self._client:
            self._client = self.create_client()

        system_content, converted_messages = self._convert_messages(messages)

        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "tools": tools,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs
            }
            if system_content:
                request_params["system"] = system_content

            response = self._client.messages.create(**request_params)
            return response

        except Exception as e:
            raise HelloAgentsException(f"Anthropic工具调用失败: {str(e)}")


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini适配器

    处理Gemini特有的API格式
    """

    def create_client(self) -> Any:
        """创建Gemini客户端"""
        try:
            import google.generativeai as genai
        except ImportError:
            raise HelloAgentsException(
                "使用Gemini需要安装: pip install google-generativeai"
            )

        genai.configure(api_key=self.api_key)
        return genai

    def _convert_messages(self, messages: List[Dict]) -> tuple[Optional[str], List[Dict]]:
        """转换消息格式"""
        system_instruction = None
        converted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                # Gemini使用 "user" 和 "model" 作为角色
                role = "model" if msg["role"] == "assistant" else "user"
                converted_messages.append({
                    "role": role,
                    "parts": [msg["content"]]
                })

        return system_instruction, converted_messages

    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            # 创建生成配置
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs.pop("max_tokens")

            # 创建模型
            model_params = {"model_name": self.model}
            if system_instruction:
                model_params["system_instruction"] = system_instruction

            model = self._client.GenerativeModel(**model_params)

            # 生成内容
            response = model.generate_content(
                converted_messages,
                generation_config=generation_config if generation_config else None
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # 提取内容
            content = response.text if hasattr(response, 'text') else ""

            # 提取usage
            usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise HelloAgentsException(f"Gemini API调用失败: {str(e)}")

    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs.pop("max_tokens")

            model_params = {"model_name": self.model}
            if system_instruction:
                model_params["system_instruction"] = system_instruction

            model = self._client.GenerativeModel(**model_params)

            usage = {}

            response = model.generate_content(
                converted_messages,
                generation_config=generation_config if generation_config else None,
                stream=True
            )

            for chunk in response:
                if hasattr(chunk, 'text'):
                    yield chunk.text

                # 尝试提取usage（可能在最后一个chunk）
                if hasattr(chunk, 'usage_metadata'):
                    usage = {
                        "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                        "completion_tokens": chunk.usage_metadata.candidates_token_count,
                        "total_tokens": chunk.usage_metadata.total_token_count,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms
            )

        except Exception as e:
            raise HelloAgentsException(f"Gemini API流式调用失败: {str(e)}")

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Any:
        """工具调用（Gemini格式）"""
        if not self._client:
            self._client = self.create_client()

        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            # 转换工具格式为Gemini格式
            gemini_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    gemini_tools.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {})
                    })

            model_params = {"model_name": self.model}
            if system_instruction:
                model_params["system_instruction"] = system_instruction

            model = self._client.GenerativeModel(**model_params, tools=gemini_tools)

            response = model.generate_content(converted_messages)
            return response

        except Exception as e:
            raise HelloAgentsException(f"Gemini工具调用失败: {str(e)}")
class DeepSeekAdapter(BaseLLMAdapter):
    """
    DeepSeek 专用适配器，支持 deepseek-v4-pro 思考模式 + 工具调用。

    deepseek-v4-pro 与 deepseek-chat 的关键差异：
    1. 需要传 extra_body={"thinking": {"type": "enabled"}} 开启思考模式
    2. 需要传 reasoning_effort="high"
    3. 工具调用轮次中，assistant 消息必须携带 reasoning_content 回传
       否则 API 返回 400
    4. assistant 消息必须原样 append（含 reasoning_content），不能手动重组
    """

    # 判断是否是需要思考模式的模型
    THINKING_MODELS = {"deepseek-v4-pro", "deepseek-reasoner"}

    def _is_thinking_model(self, model: str) -> bool:
        return model.lower() in self.THINKING_MODELS

    def _thinking_kwargs(self) -> dict:
        """返回思考模式所需的额外参数"""
        if self._is_thinking_model(self.model):
            return {
                "reasoning_effort": "high",
                "extra_body": {"thinking": {"type": "enabled"}},
            }
        return {}

    def create_client(self) -> Any:
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def create_async_client(self) -> Any:
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ── 非流式调用 ──────────────────────────────────────────
    def invoke(self, messages: List[Dict], **kwargs) -> LLMResponse:
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        merged = {**self._thinking_kwargs(), **kwargs}

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **merged,
            )
            latency_ms = int((time.time() - start_time) * 1000)
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning_content = (
                getattr(choice.message, "reasoning_content", None)
                or getattr(choice, "reasoning_content", None)
            )
            usage = {}
            if getattr(response, "usage", None):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content,
            )
        except Exception as e:
            raise HelloAgentsException(f"DeepSeek API调用失败: {e}")

    # ── 同步流式调用 ────────────────────────────────────────
    def stream_invoke(self, messages: List[Dict], **kwargs) -> Iterator[str]:
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        merged = {**self._thinking_kwargs(), **kwargs}

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **merged,
            )
            reasoning_content = None
            usage = {}

            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        yield delta.content
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        reasoning_content = (reasoning_content or "") + rc
                if getattr(chunk, "usage", None):
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=int((time.time() - start_time) * 1000),
                reasoning_content=reasoning_content,
            )
        except Exception as e:
            raise HelloAgentsException(f"DeepSeek 流式调用失败: {e}")

    # ── 异步流式调用 ────────────────────────────────────────
    async def astream_invoke(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        if not self._async_client:
            self._async_client = self.create_async_client()

        start_time = time.time()
        merged = {**self._thinking_kwargs(), **kwargs}

        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **merged,
            )
            reasoning_content = None
            usage = {}

            async for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        yield delta.content
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        reasoning_content = (reasoning_content or "") + rc
                if getattr(chunk, "usage", None):
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=int((time.time() - start_time) * 1000),
                reasoning_content=reasoning_content,
            )
        except Exception as e:
            raise HelloAgentsException(f"DeepSeek 异步流式调用失败: {e}")

    # ── 同步工具调用 ────────────────────────────────────────
    def invoke_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: Union[str, Dict] = "auto",
        **kwargs,
    ) -> Any:
        """
        deepseek-v4-pro 工具调用的核心修复：

        1. 自动注入思考模式参数
        2. assistant 消息中的 reasoning_content 必须原样回传
           （react_agent 手动重组消息会丢掉它，这里在清洗时补回去）
        3. tool_choice 默认 auto（required 会阻止 Finish 输出）
        """
        if not self._client:
            self._client = self.create_client()

        cleaned_messages = self._clean_messages(messages)
        merged = {**self._thinking_kwargs(), **kwargs}

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": cleaned_messages,
            **merged,
        }
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = tool_choice

        try:
            return self._client.chat.completions.create(**call_kwargs)
        except Exception as e:
            raise HelloAgentsException(f"DeepSeek Function Calling 失败: {e}")

    # ── 异步工具调用 ────────────────────────────────────────
    async def ainvoke_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: Union[str, Dict] = "auto",
        **kwargs,
    ) -> Any:
        if not self._async_client:
            self._async_client = self.create_async_client()

        cleaned_messages = self._clean_messages(messages)
        merged = {**self._thinking_kwargs(), **kwargs}

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": cleaned_messages,
            **merged,
        }
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = tool_choice

        try:
            return await self._async_client.chat.completions.create(**call_kwargs)
        except Exception as e:
            raise HelloAgentsException(f"DeepSeek 异步 Function Calling 失败: {e}")

    # ── 异步流式工具调用 ────────────────────────────────────
    async def astream_invoke_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: Union[str, Dict] = "auto",
        **kwargs,
    ):
        """
        DeepSeek 真流式工具调用。

        特殊处理：
        - 注入 thinking 参数（deepseek-v4-pro）
        - 清洗消息（回传 reasoning_content，防 400）
        - 收集 reasoning_content delta 并实时 yield
        - 收集 tool_calls delta 拼接完整列表

        Yields:
            {"type": "text",       "content": str}
            {"type": "thinking",   "content": str}
            {"type": "tool_calls", "tool_calls": list}   # 完整列表，流结束后
            {"type": "done",       "response": None}
        """
        if not self._async_client:
            self._async_client = self.create_async_client()

        cleaned_messages = self._clean_messages(messages)
        merged = {**self._thinking_kwargs(), **kwargs}

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": cleaned_messages,
            "stream": True,
            **merged,
        }
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = tool_choice

        try:
            stream = await self._async_client.chat.completions.create(**call_kwargs)

            tool_calls_map: Dict[int, Dict] = {}

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # 文本 delta
                if getattr(delta, "content", None):
                    yield {"type": "text", "content": delta.content}

                # 推理 delta（deepseek-v4-pro reasoning_content）
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    yield {"type": "thinking", "content": rc}

                # tool_calls delta
                if getattr(delta, "tool_calls", None):
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        entry = tool_calls_map[idx]
                        if tc_delta.id:
                            entry["id"] += tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments

            if tool_calls_map:
                tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)]
                yield {"type": "tool_calls", "tool_calls": tool_calls}

            yield {"type": "done", "response": None}

        except Exception as e:
            raise HelloAgentsException(f"DeepSeek 流式 Function Calling 失败: {e}")

    # ── 消息清洗（核心）────────────────────────────────────
    def _clean_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        清洗消息列表，确保符合 deepseek-v4-pro 协议：

        - assistant 消息：保留 content、tool_calls、reasoning_content
          （工具调用轮次中 reasoning_content 必须回传，否则 400）
        - tool 消息：保留 content、tool_call_id
        - system/user 消息：只保留 content
        - 过滤非法 role
        """
        cleaned = []
        for msg in messages:
            # 兼容 OpenAI SDK 返回的对象（非 dict）
            if not isinstance(msg, dict):
                # 直接原样 append SDK 对象（最安全，API 能识别）
                cleaned.append(msg)
                continue

            role = msg.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                continue

            content = msg.get("content") or ""

            if role == "assistant":
                safe: Dict[str, Any] = {"role": "assistant", "content": content}
                # ★ 保留 tool_calls
                if msg.get("tool_calls"):
                    safe["tool_calls"] = msg["tool_calls"]
                # ★ 保留 reasoning_content（deepseek-v4-pro 工具调用轮次必须回传）
                # thinking model 要求每条 assistant 消息都必须带该字段，
                # 即使本轮没有推理内容，也必须传空字符串，否则 API 返回 400
                if self._is_thinking_model(self.model):
                    safe["reasoning_content"] = msg.get("reasoning_content") or ""
                elif msg.get("reasoning_content"):
                    safe["reasoning_content"] = msg["reasoning_content"]
            elif role == "tool":
                safe = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id", ""),
                }
            else:
                safe = {"role": role, "content": content}

            cleaned.append(safe)
        return cleaned


def create_adapter(
    api_key: str,
    base_url: Optional[str],
    timeout: int,
    model: str,
) -> BaseLLMAdapter:
    if base_url:
        url = base_url.lower()
        if "anthropic.com" in url:
            return AnthropicAdapter(api_key, base_url, timeout, model)
        if "googleapis.com" in url or "generativelanguage" in url:
            return GeminiAdapter(api_key, base_url, timeout, model)
        if "deepseek.com" in url:
            return DeepSeekAdapter(api_key, base_url, timeout, model)
    return OpenAIAdapter(api_key, base_url, timeout, model)
