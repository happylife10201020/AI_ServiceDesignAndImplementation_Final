"""OpenAI Chat/Embedding 모델 생성."""

from __future__ import annotations

from config import Settings


def get_chat_model(settings: Settings, *, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.chat_model,
        temperature=temperature,
        api_key=settings.api_key,
    )


def get_embeddings(settings: Settings):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.api_key)
