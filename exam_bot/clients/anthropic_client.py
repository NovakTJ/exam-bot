from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from anthropic import Anthropic


class AnthropicTextConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, description="Anthropic API key; if None, will use env ANTHROPIC_API_KEY")
    model: str = Field(default="claude-3-7-sonnet-latest")
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.0)


class TextResponse(BaseModel):
    model: str
    content: str
    stop_reason: Optional[str] = None


class AnthropicTextClient:
    """Synchronous Messages API wrapper.

    Contract:
    - input: prompt (str), optional context (str)
    - behavior: sends a single user message (prompt + optional context)
    - output: concatenated text content returned by Claude
    """

    def __init__(self, config: Optional[AnthropicTextConfig] = None):
        self.config = config or AnthropicTextConfig()
        if self.config.api_key:
            self.client = Anthropic(api_key=self.config.api_key)
        else:
            self.client = Anthropic()

    def run(self, prompt: str, context: Optional[str] = None, *, model: Optional[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> TextResponse:
        model_name = model or self.config.model
        max_toks = max_tokens if max_tokens is not None else self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        text = prompt if not context else f"{prompt}\n\nContext:\n{context}"

        msg = self.client.messages.create(
            model=model_name,
            max_tokens=max_toks,
            temperature=temp,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ],
        )

        chunks = []
        for item in (msg.content or []):
            if getattr(item, "type", None) == "text":
                # SDK >=0.30 returns objects with .text; older returns dicts
                chunks.append(getattr(item, "text", "") or item.get("text", ""))
        return TextResponse(model=msg.model, content="\n".join(chunks).strip(), stop_reason=getattr(msg, "stop_reason", None))
