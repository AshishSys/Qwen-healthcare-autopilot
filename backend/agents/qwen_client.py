"""
Qwen Cloud Client — Alibaba Cloud Model Studio (DashScope) Integration

This module provides the core LLM interface for the Healthcare Autopilot system.
It connects to Qwen Cloud via the DashScope API and implements:
- Model routing (qwen-max for complex reasoning, qwen-turbo for fast triage)
- Function calling for agent tool use
- Streaming support for real-time responses
- Token usage tracking and cost monitoring
"""

import os
import json
import logging
from typing import Optional
from enum import Enum
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)


class QwenModel(str, Enum):
    """Available Qwen models on Alibaba Cloud Model Studio."""
    MAX = "qwen-max"           # Complex clinical reasoning, multi-step analysis
    PLUS = "qwen-plus"         # Standard workflows, balanced quality/cost
    TURBO = "qwen-turbo"       # Fast triage, simple routing decisions
    LONG = "qwen-long"         # Long context for document analysis
    VL_MAX = "qwen-vl-max"    # Vision: medical imaging analysis


@dataclass
class TokenUsage:
    """Track token consumption for cost monitoring."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, usage):
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens


@dataclass 
class AgentResponse:
    """Structured response from the Qwen agent."""
    content: str
    tool_calls: list = field(default_factory=list)
    finish_reason: str = ""
    usage: Optional[TokenUsage] = None
    model: str = ""


class QwenCloudClient:
    """
    Client for Alibaba Cloud Model Studio (DashScope) — Qwen Cloud.
    
    Uses OpenAI-compatible API endpoint provided by DashScope,
    enabling seamless integration with existing tooling while
    leveraging Qwen's advanced capabilities.
    
    Environment Variables:
        DASHSCOPE_API_KEY: Your Alibaba Cloud DashScope API key
        QWEN_BASE_URL: API endpoint (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
    """
    
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # Model routing based on task complexity
    TASK_MODEL_MAP = {
        "triage": QwenModel.MAX,
        "lab_interpretation": QwenModel.MAX,
        "care_plan": QwenModel.MAX,
        "prior_auth": QwenModel.MAX,
        "patient_communication": QwenModel.PLUS,
        "scheduling": QwenModel.TURBO,
        "summarization": QwenModel.PLUS,
        "simple_routing": QwenModel.TURBO,
    }
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv("QWEN_BASE_URL", self.DEFAULT_BASE_URL)
        
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY not set. Get your key from "
                "https://dashscope.console.aliyun.com/apiKey"
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        self.session_usage = TokenUsage()
        logger.info(f"QwenCloudClient initialized with endpoint: {self.base_url}")
    
    def get_model_for_task(self, task_type: str) -> str:
        """Route to optimal model based on task complexity."""
        model = self.TASK_MODEL_MAP.get(task_type, QwenModel.PLUS)
        logger.info(f"Task '{task_type}' routed to model: {model}")
        return model
    
    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> AgentResponse:
        """
        Send a chat completion request to Qwen Cloud.
        
        Args:
            messages: Conversation history in OpenAI format
            model: Specific model override (or use task_type routing)
            task_type: Task category for automatic model selection
            tools: Function definitions for tool calling
            temperature: Sampling temperature (low for clinical accuracy)
            max_tokens: Maximum response tokens
            stream: Enable streaming response
            
        Returns:
            AgentResponse with content, tool_calls, and usage info
        """
        resolved_model = model or self.get_model_for_task(task_type or "default")
        
        kwargs = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            if stream:
                return self._stream_chat(**kwargs)
            
            response = self.client.chat.completions.create(**kwargs)
            
            choice = response.choices[0]
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
            self.session_usage.add(usage)
            
            # Extract tool calls if present
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                        }
                    })
            
            return AgentResponse(
                content=choice.message.content or "",
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                usage=usage,
                model=resolved_model,
            )
            
        except Exception as e:
            logger.error(f"Qwen Cloud API error: {e}")
            raise
    
    def _stream_chat(self, **kwargs) -> AgentResponse:
        """Handle streaming responses for real-time UI updates."""
        chunks = []
        tool_calls = []
        
        stream = self.client.chat.completions.create(**kwargs)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
            if chunk.choices[0].delta.tool_calls:
                for tc in chunk.choices[0].delta.tool_calls:
                    tool_calls.append(tc)
        
        return AgentResponse(
            content="".join(chunks),
            tool_calls=tool_calls,
            finish_reason="stop",
            model=kwargs.get("model", ""),
        )
    
    def function_call_loop(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: callable,
        task_type: str = "default",
        max_iterations: int = 10,
    ) -> AgentResponse:
        """
        Execute a full agent loop with tool calling until completion.
        
        This is the core autonomous agent pattern:
        1. Send messages + tools to Qwen
        2. If Qwen returns tool_calls, execute them
        3. Append tool results and loop back to step 1
        4. Continue until Qwen returns a final text response
        
        Args:
            messages: Initial conversation
            tools: Available tool definitions
            tool_executor: Function that executes tool calls and returns results
            task_type: For model routing
            max_iterations: Safety limit on loops
            
        Returns:
            Final AgentResponse after all tool calls complete
        """
        iteration = 0
        current_messages = list(messages)
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent loop iteration {iteration}/{max_iterations}")
            
            response = self.chat(
                messages=current_messages,
                tools=tools,
                task_type=task_type,
            )
            
            # No tool calls — agent is done
            if not response.tool_calls:
                logger.info(f"Agent completed after {iteration} iterations")
                return response
            
            # Execute tool calls and append results
            current_messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": json.dumps(tc["function"]["arguments"]),
                        }
                    }
                    for tc in response.tool_calls
                ]
            })
            
            for tool_call in response.tool_calls:
                result = tool_executor(
                    tool_call["function"]["name"],
                    tool_call["function"]["arguments"],
                )
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result) if isinstance(result, dict) else str(result),
                })
        
        logger.warning(f"Agent hit max iterations ({max_iterations})")
        return response
