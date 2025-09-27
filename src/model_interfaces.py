"""
Model Interface Implementations for Theory of Mind Benchmark

This module provides standardized interfaces for evaluating different LLMs
on the Theory of Mind benchmark, supporting multiple API providers.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model evaluation."""
    model_name: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class BaseModelInterface(ABC):
    """Abstract base class for model interfaces."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.request_count = 0
        self.total_tokens = 0

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate response from model."""
        pass

    def __call__(self, prompt: str) -> str:
        """Make the interface callable."""
        return self.generate_response(prompt)

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "model_name": self.config.model_name,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "average_tokens_per_request": self.total_tokens / max(self.request_count, 1)
        }


class OpenAIInterface(BaseModelInterface):
    """Interface for OpenAI models (GPT-3.5, GPT-4, etc.)."""

    def __init__(self, config: ModelConfig, api_key: Optional[str] = None):
        super().__init__(config)

        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        self.client = openai.OpenAI(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty,
                    timeout=self.config.timeout
                )

                self.request_count += 1
                self.total_tokens += response.usage.total_tokens

                return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.config.retry_attempts}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise e

        return "Error: Failed to generate response after all retry attempts"


class AnthropicInterface(BaseModelInterface):
    """Interface for Anthropic Claude models."""

    def __init__(self, config: ModelConfig, api_key: Optional[str] = None):
        super().__init__(config)

        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        """Generate response using Anthropic API."""
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    messages=[{"role": "user", "content": prompt}]
                )

                self.request_count += 1
                # Anthropic doesn't provide token usage in the same way, estimate
                self.total_tokens += len(prompt.split()) + len(response.content[0].text.split())

                return response.content[0].text

            except Exception as e:
                logger.warning(f"Anthropic API error (attempt {attempt + 1}/{self.config.retry_attempts}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise e

        return "Error: Failed to generate response after all retry attempts"


class HuggingFaceInterface(BaseModelInterface):
    """Interface for Hugging Face models."""

    def __init__(self, config: ModelConfig, api_key: Optional[str] = None, use_local: bool = False):
        super().__init__(config)
        self.use_local = use_local

        if use_local:
            self._setup_local_model()
        else:
            self._setup_api_model(api_key)

    def _setup_local_model(self):
        """Setup local Hugging Face model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

        except ImportError:
            raise ImportError("Transformers package not installed. Run: pip install transformers torch")

    def _setup_api_model(self, api_key: Optional[str]):
        """Setup Hugging Face API model."""
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Requests package not installed. Run: pip install requests")

        api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("Hugging Face API key not provided. Set HUGGINGFACE_API_KEY environment variable.")

        self.api_key = api_key
        self.api_url = f"https://api-inference.huggingface.co/models/{self.config.model_name}"

    def generate_response(self, prompt: str) -> str:
        """Generate response using Hugging Face model."""
        if self.use_local:
            return self._generate_local_response(prompt)
        else:
            return self._generate_api_response(prompt)

    def _generate_local_response(self, prompt: str) -> str:
        """Generate response using local model."""
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            self.request_count += 1

            # Extract generated text (remove the input prompt)
            generated_text = outputs[0]['generated_text']
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()

            return response

        except Exception as e:
            logger.error(f"Local model generation error: {e}")
            return f"Error: Failed to generate response - {e}"

    def _generate_api_response(self, prompt: str) -> str:
        """Generate response using Hugging Face API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(self.config.retry_attempts):
            try:
                response = self.requests.post(
                    self.api_url,
                    headers=headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": self.config.max_tokens,
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "do_sample": True
                        }
                    },
                    timeout=self.config.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    self.request_count += 1

                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        # Remove input prompt from response
                        if generated_text.startswith(prompt):
                            return generated_text[len(prompt):].strip()
                        return generated_text.strip()
                    else:
                        return "Error: Unexpected API response format"

                else:
                    logger.warning(f"HF API error {response.status_code}: {response.text}")

            except Exception as e:
                logger.warning(f"HF API error (attempt {attempt + 1}/{self.config.retry_attempts}): {e}")

            if attempt < self.config.retry_attempts - 1:
                time.sleep(self.config.retry_delay * (2 ** attempt))

        return "Error: Failed to generate response after all retry attempts"


class MockModelInterface(BaseModelInterface):
    """Mock interface for testing and demonstration."""

    def __init__(self, config: ModelConfig, response_pattern: str = "random"):
        super().__init__(config)
        self.response_pattern = response_pattern

    def generate_response(self, prompt: str) -> str:
        """Generate mock response."""
        import random

        self.request_count += 1

        if "Theory of Mind" in prompt or "scenario" in prompt.lower():
            # Mock ToM-style responses
            if self.response_pattern == "always_correct":
                # Extract correct answer from prompt if visible
                for letter in ['A', 'B', 'C', 'D', 'E']:
                    if f"{letter}." in prompt:
                        return f"{letter}. This is the correct answer based on theory of mind reasoning."
                return "A. Default correct answer."

            elif self.response_pattern == "random":
                answer = random.choice(['A', 'B', 'C', 'D', 'E'])
                reasoning = [
                    "Based on the social dynamics described",
                    "Considering the mental states involved",
                    "Given the perspective-taking required",
                    "Analyzing the beliefs and intentions"
                ]
                return f"{answer}. {random.choice(reasoning)}, this seems most appropriate."

            elif self.response_pattern == "poor_tom":
                # Simulate poor theory of mind understanding
                return "A. People always say what they think directly."

        return "This is a mock response for testing purposes."


def create_model_interface(model_name: str,
                         provider: str = "auto",
                         config: Optional[ModelConfig] = None,
                         **kwargs) -> BaseModelInterface:
    """
    Factory function to create appropriate model interface.

    Args:
        model_name: Name of the model to use
        provider: API provider ("openai", "anthropic", "huggingface", "mock", or "auto")
        config: Optional ModelConfig object
        **kwargs: Additional arguments for the interface

    Returns:
        Configured model interface
    """
    if config is None:
        config = ModelConfig(model_name=model_name)

    # Auto-detect provider based on model name
    if provider == "auto":
        if "gpt" in model_name.lower():
            provider = "openai"
        elif "claude" in model_name.lower():
            provider = "anthropic"
        elif "mock" in model_name.lower():
            provider = "mock"
        else:
            provider = "huggingface"

    provider = provider.lower()

    if provider == "openai":
        return OpenAIInterface(config, **kwargs)
    elif provider == "anthropic":
        return AnthropicInterface(config, **kwargs)
    elif provider == "huggingface":
        return HuggingFaceInterface(config, **kwargs)
    elif provider == "mock":
        return MockModelInterface(config, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Predefined model configurations
PRESET_MODELS = {
    "gpt-4": ModelConfig(
        model_name="gpt-4",
        max_tokens=1000,
        temperature=0.3,  # Lower temperature for more consistent reasoning
    ),
    "gpt-3.5-turbo": ModelConfig(
        model_name="gpt-3.5-turbo",
        max_tokens=800,
        temperature=0.3,
    ),
    "claude-3-sonnet": ModelConfig(
        model_name="claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0.3,
    ),
    "claude-3-haiku": ModelConfig(
        model_name="claude-3-haiku-20240307",
        max_tokens=800,
        temperature=0.3,
    ),
    "llama-2-7b": ModelConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        max_tokens=512,
        temperature=0.3,
    ),
    "mock-random": ModelConfig(
        model_name="mock-random",
        max_tokens=200,
        temperature=0.0,
    ),
    "mock-correct": ModelConfig(
        model_name="mock-always-correct",
        max_tokens=200,
        temperature=0.0,
    )
}


def get_preset_model(model_key: str, **kwargs) -> BaseModelInterface:
    """
    Get a preconfigured model interface.

    Args:
        model_key: Key from PRESET_MODELS
        **kwargs: Override arguments for the interface

    Returns:
        Configured model interface
    """
    if model_key not in PRESET_MODELS:
        raise ValueError(f"Unknown preset model: {model_key}. Available: {list(PRESET_MODELS.keys())}")

    config = PRESET_MODELS[model_key]

    # Special handling for mock models
    if "mock" in model_key:
        pattern = "always_correct" if "correct" in model_key else "random"
        return MockModelInterface(config, response_pattern=pattern)

    return create_model_interface(config.model_name, config=config, **kwargs)