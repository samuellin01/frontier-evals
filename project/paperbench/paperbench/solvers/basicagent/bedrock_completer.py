from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Unpack

import anthropic
import structlog.stdlib
import tenacity
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic import Field
from typing_extensions import override

from paperbench.solvers.basicagent.completer import (
    BasicAgentTurnCompleterConfig,
    TimeTrackingRetryConfig,
)
from paperbench.solvers.basicagent.tools.base import Tool
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import RetryConfig

logger = structlog.stdlib.get_logger(component=__name__)

BEDROCK_RETRY_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


def _convert_oai_messages_to_claude(
    messages: list[ChatCompletionMessageParam],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_prompt = None
    claude_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]

        if role in ("system", "developer"):
            text = msg.get("content", "")
            if isinstance(text, list):
                text = "\n".join(
                    p.get("text", "") for p in text if isinstance(p, dict) and p.get("type") == "text"
                )
            if system_prompt is None:
                system_prompt = str(text)
            else:
                system_prompt += "\n\n" + str(text)

        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                claude_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append({"type": "text", "text": p["text"]})
                claude_messages.append({"role": "user", "content": parts or ""})

        elif role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            text_content = msg.get("content")
            if text_content:
                if isinstance(text_content, str):
                    content_blocks.append({"type": "text", "text": text_content})
                elif isinstance(text_content, list):
                    for p in text_content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            content_blocks.append({"type": "text", "text": p["text"]})

            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
                args_str = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"raw": args_str}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc_id,
                    "name": name,
                    "input": args,
                })

            if content_blocks:
                claude_messages.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            claude_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(content),
                }],
            })

    # Merge consecutive same-role messages (Claude requires alternating roles)
    merged: list[dict[str, Any]] = []
    for msg in claude_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev_content = merged[-1]["content"]
            new_content = msg["content"]
            if isinstance(prev_content, str):
                prev_content = [{"type": "text", "text": prev_content}]
            if isinstance(new_content, str):
                new_content = [{"type": "text", "text": new_content}]
            merged[-1]["content"] = prev_content + new_content
        else:
            merged.append(msg)

    return system_prompt, merged


def _convert_oai_tools_to_claude(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claude_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            claude_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
    return claude_tools


def _convert_claude_response_to_oai(
    response: anthropic.types.Message,
) -> list[ChatCompletionMessage]:
    text_parts: list[str] = []
    tool_calls: list[ChatCompletionMessageFunctionToolCall] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=block.id,
                    type="function",
                    function=Function(
                        name=block.name,
                        arguments=json.dumps(block.input),
                    ),
                )
            )

    content = "\n".join(text_parts) if text_parts else None

    return [
        ChatCompletionMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )
    ]


class BedrockClaudeTurnCompleter(TurnCompleter):
    def __init__(
        self,
        model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
        aws_region: str = "us-east-1",
        max_tokens: int = 16384,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self.model = model
        self.aws_region = aws_region
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = tools or []
        self.retry_config = retry_config or TimeTrackingRetryConfig()
        self.encoding_name = "cl100k_base"
        self.n_ctx = 200000

        self._client = anthropic.AnthropicBedrock(aws_region=aws_region)

    @override
    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> TurnCompleter.Completion:
        raise NotImplementedError("Use async_completion instead")

    @override
    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> TurnCompleter.Completion:
        system_prompt, claude_messages = _convert_oai_messages_to_claude(conversation)
        claude_tools = _convert_oai_tools_to_claude(self.tools)

        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": claude_messages,
            "max_tokens": self.max_tokens,
        }
        if system_prompt:
            api_kwargs["system"] = system_prompt
        if claude_tools:
            api_kwargs["tools"] = claude_tools
        if self.temperature is not None:
            api_kwargs["temperature"] = self.temperature

        retry = self.retry_config.build()
        async for attempt in retry:
            with attempt:
                response = self._client.messages.create(**api_kwargs)

        output_messages = _convert_claude_response_to_oai(response)

        return TurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=output_messages,
        )

    class Config(TurnCompleter.Config):
        model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        aws_region: str = "us-east-1"
        max_tokens: int = 16384
        temperature: float | None = None

        @override
        def build(self) -> BedrockClaudeTurnCompleter:
            return BedrockClaudeTurnCompleter(
                model=self.model,
                aws_region=self.aws_region,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )


class BedrockClaudeTurnCompleterConfig(
    BedrockClaudeTurnCompleter.Config, BasicAgentTurnCompleterConfig
):
    @override
    def build(self) -> BedrockClaudeTurnCompleter:
        tools: list[dict[str, Any]] = []
        if self.basicagent_tools is not None:
            for tool in self.basicagent_tools:
                oai_tool = tool.get_oai_tool_call()
                tools.append({
                    "type": "function",
                    "function": {
                        "name": oai_tool["name"],
                        "description": oai_tool.get("description", ""),
                        "parameters": oai_tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                })

        return BedrockClaudeTurnCompleter(
            model=self.model,
            aws_region=self.aws_region,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=tools,
            retry_config=self.retry_config,
        )
