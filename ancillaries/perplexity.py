import json
import os
from typing import Any, Dict, List

import requests
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

from .wasm import WasmChatService


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


class PerplexityChatService(WasmChatService):
    # model: str = "mistral-7b-instruct"
    model: str = "pplx-70b-chat"
    # model: str = "llama-2-70b-chat"
    # model: str = "pplx-70b-online"
    # model: str = "mixtral-8x7b-instruct"
    service_url: str = "https://api.perplexity.ai/chat/completions"
    api_key: str | None = None

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        payload = {
            "model": self.model,
            "messages": [_convert_message_to_dict(m) for m in messages],
        }

        res = requests.post(
            url=self.service_url,
            timeout=self.request_timeout,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f'Bearer {os.getenv("PERPLEXITYAI_API_KEY")}',
            },
            data=json.dumps(payload),
        )
        return res
