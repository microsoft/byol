# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Embedding clients for semantic similarity computation.

This module provides a unified interface for embedding models:
- Azure OpenAI text-embedding-3-large
- Qwen3-Embedding-8B via HuggingFace transformers
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch
from dotenv import load_dotenv


# Type alias for embedding model choices
EmbeddingModelType = Literal["openai", "qwen"]


class EmbeddingClient:
    """
    Unified interface for embedding models.
    
    Supports:
    - Azure OpenAI text-embedding-3-large
    - Qwen3-Embedding-8B via HuggingFace transformers
    
    Usage:
        client = EmbeddingClient(model_type="openai")
        embeddings = client.embed(["Hello world", "Goodbye world"])
    """
    
    def __init__(self, model_type: EmbeddingModelType = "openai", device: str = "cuda:0"):
        """
        Initialize the embedding client.
        
        Args:
            model_type: Either "openai" or "qwen"
            device: Device for tensor operations (used by Qwen)
        """
        self.model_type = model_type
        self.device = device
        self._client = None
        self._deployment = None
        self._qwen_model = None
        self._qwen_tokenizer = None
        
        if model_type == "openai":
            self._init_openai()
        elif model_type == "qwen":
            self._init_qwen()
        else:
            raise ValueError(f"Unknown embedding model type: {model_type}")
    
    def _init_openai(self) -> None:
        """Initialize Azure OpenAI embedding client."""
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AzureOpenAI
        
        load_dotenv()
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT environment variable not set.\n"
                "Please set it to your Azure OpenAI endpoint URL."
            )
        
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        
        self._client = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
        )
        self._deployment = "text-embedding-3-large"
        print(f"Initialized Azure OpenAI embedding model: {self._deployment}")
    
    def _init_qwen(self) -> None:
        """Initialize Qwen embedding model via HuggingFace transformers."""
        from transformers import AutoModel, AutoTokenizer
        
        model_name = "Qwen/Qwen3-Embedding-8B"
        print(f"Loading Qwen embedding model: {model_name}")
        
        self._qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._qwen_model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._qwen_model.eval()
        
        print(f"Initialized Qwen embedding model: {model_name}")
    
    def embed(self, texts: list[str]) -> torch.Tensor:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tensor of shape (len(texts), embedding_dim)
        """
        if self.model_type == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_qwen(texts)
    
    def _embed_openai(self, texts: list[str]) -> torch.Tensor:
        """Get embeddings using Azure OpenAI."""
        response = self._client.embeddings.create(input=texts, model=self._deployment)
        embeddings_array = np.array([e.embedding for e in response.data])
        # Keep on CPU - similarity will be computed on CPU
        return torch.tensor(embeddings_array, dtype=torch.float32)
    
    def _embed_qwen(self, texts: list[str]) -> torch.Tensor:
        """Get embeddings using Qwen via HuggingFace transformers."""
        # Tokenize
        inputs = self._qwen_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self._qwen_model.device) for k, v in inputs.items()}
        
        # Get embeddings (mean pooling over last hidden state)
        with torch.no_grad():
            outputs = self._qwen_model(**inputs)
            # Use last_hidden_state and mean pool
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Move to CPU immediately to free GPU memory
            embeddings = embeddings.detach().cpu().float()
        
        # Delete intermediates to free GPU memory
        del outputs
        del inputs
        torch.cuda.empty_cache()
        
        return embeddings
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache (useful for Qwen model)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Legacy function for backward compatibility
def get_embedding_client(
    model_type: EmbeddingModelType = "openai",
    device: str = "cuda:0",
) -> EmbeddingClient:
    """
    Create an embedding client.
    
    Args:
        model_type: Either "openai" or "qwen"
        device: Device for tensor operations
        
    Returns:
        Configured EmbeddingClient instance
    """
    return EmbeddingClient(model_type=model_type, device=device)


__all__ = [
    "EmbeddingModelType",
    "EmbeddingClient",
    "get_embedding_client",
]
