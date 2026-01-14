# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI API client wrapper with tool integration."""

import os

import openai
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class OpenAIProvider(ProviderConfig):
    """OpenAI provider configuration."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create OpenAI client with base URL."""
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "OpenAI"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "openai"

    def get_extra_headers(self) -> dict[str, str]:
        """Get OpenAI-specific headers."""
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        return True


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI client wrapper using the standard OpenAI API."""

    def __init__(self, model_config: ModelConfig):
        provider_config = OpenAIProvider()
        super().__init__(model_config, provider_config)
