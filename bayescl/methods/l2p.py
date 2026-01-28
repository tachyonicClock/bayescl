# https://github.com/ContinualAI/avalanche/blob/master/avalanche/models/prompt.py
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.training.plugins import SupervisedPlugin
from torch import Tensor


class Backbone(ABC):
    """Abstract interface for vision transformer backbones used in L2P."""

    embed_dim: int  # Embedding dimension of the backbone

    @abstractmethod
    def embed_patches(self, pixel_values: Tensor) -> Tensor:
        """Turn image pixel values into patch embeddings.

        :param pixel_values: A tensor image of shape (batch_size, channels, height, width)
        :return: A tensor of shape (batch_size, num_patches, embed_dim)
        """

    @abstractmethod
    def forward_encoder(self, patch_embeddings: Tensor) -> Tensor:
        """Get final features from patch embeddings.

        :param patch_embeddings: A tensor of shape (batch_size, num_patches, embed_dim)
        :return: A tensor of shape (batch_size, num_patches, embed_dim)
        """

    @abstractmethod
    def forward_query(self, patch_embeddings: Tensor) -> Tensor:
        """Get query features from patch embeddings for prompt selection.

        :param patch_embeddings: A tensor of shape (batch_size, num_patches, embed_dim)
        :return: A tensor of shape (batch_size, embed_dim)
        """


class PromptPool(nn.Module):
    def __init__(
        self, pool_size: int, prompt_length: int, embed_dim: int, top_k: int
    ) -> None:
        """Learning 2 Prompt Pool module.

        :param pool_size: Number of prompts in the pool
        :param prompt_length: Length of each prompt (number of tokens/patches)
        :param embed_dim: Dimension of the embeddings
        :param top_k: Number of top prompts to retrieve per query
        """
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.keys = nn.Parameter(torch.empty(pool_size, embed_dim))
        self.prompts = nn.Parameter(torch.empty(pool_size, prompt_length, embed_dim))

        nn.init.uniform_(self.prompts, -1, 1)
        nn.init.uniform_(self.keys, -1, 1)

    @staticmethod
    def lookup(
        query: Tensor, key: Tensor, prompt_pool: Tensor, top_k: int
    ) -> Tuple[Tensor, Tensor]:
        """Lookup prompts based on cosine similarity between query and keys.

        :param query: Embedding to lookup (batch_size, embed_dim).
        :param key: Keys to compare against (pool_size, embed_dim).
        :param prompt_pool: Values to select (pool_size, prompt_length, embed_dim).
        :param top_k: Number of top prompts to retrieve per query.
        :return: A tuple of selected similarity scores and concatenated prompts.
        """
        if not (query.size(1) == key.size(1) == prompt_pool.size(2)):
            raise ValueError(
                "`query`, `key` and `prompt_pool` must have the same embedding size."
            )
        if not (key.size(0) == prompt_pool.size(0)):
            raise ValueError("`key` and `prompt_pool` must have the same pool size.")

        prompt_length = prompt_pool.size(1)
        embed_dim = prompt_pool.size(2)
        batch_size = query.size(0)

        # Cosine similarity between each query and each key
        key_norm = F.normalize(key, dim=1)
        query_norm = F.normalize(query, dim=1)
        similarity = query_norm @ key_norm.t()  # (batch, pool_size)

        # Get top-k most similar keys per query then retrieve corresponding prompts
        _, idx = torch.topk(similarity, top_k, dim=1)  # (batch, top_k)
        prompts = prompt_pool[idx].view(batch_size, top_k * prompt_length, embed_dim)

        # Surrogate loss to pull selected keys closer together
        similarity_loss = torch.einsum("bkd,bd->", key_norm[idx], query_norm)
        similarity_loss = similarity_loss / batch_size

        # Return prompts and sum of similarity scores for selected keys
        return similarity_loss, prompts

    def forward(self, query: Tensor) -> Tuple[Tensor, Tensor]:
        return self.lookup(
            query=query,
            key=self.keys,
            prompt_pool=self.prompts,
            top_k=self.top_k,
        )


class L2PModel(nn.Module):
    loss_: Tensor

    def __init__(
        self,
        backbone: Backbone,
        num_classes: int,
        pull_constraint_coeff: float,
        prompt_pool: PromptPool,
    ) -> None:
        """Learning 2 Prompt Module.

        :param backbone: A backbone ViT model implementing the Backbone interface.
        :param prompt_pool: A prompt pool module for retrieving prompts.
        :param pull_constraint_coeff: Weight for the prompt loss term.
        :param head: Classification head module.
        """
        super().__init__()
        if backbone.embed_dim != prompt_pool.embed_dim:
            raise ValueError("Backbone and PromptPool embed_dim must match.")

        self.backbone = backbone
        self.head: nn.Module = nn.Linear(prompt_pool.embed_dim, num_classes)
        self.prompt_pool: PromptPool = prompt_pool
        self.pull_constraint_coeff = pull_constraint_coeff

        # Freeze backbone parameters to prevent catastrophic forgetting
        if isinstance(self.backbone, nn.Module):
            for param in self.backbone.parameters():
                param.requires_grad = False
        if isinstance(self.head, nn.Module):
            for param in self.head.parameters():
                param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with prompt querying and learning.

        :param x: A tensor image of shape (batch_size, channels, height, width)
        :return: A tensor of shape (batch_size, num_classes)
        """
        # Convert images to patch embeddings
        patches = self.backbone.embed_patches(x)  # (batch_size, num_patches, embed_dim)

        # Generate query for prompt selection
        query = self.backbone.forward_query(patches)  # (batch_size, embed_dim)

        # Retrieve relevant prompts from the pool
        loss, prompts = self.prompt_pool.forward(query)

        # Eq 5: Suurogate loss to pull selected keys closer together. Cosine similarity
        # in [0, 1]
        self.loss_ = self.pull_constraint_coeff * -loss

        # Prepend selected prompts to input sequence
        # Separate CLS token from other patches
        cls_patch = patches[:, :1, :]  # (batch_size, 1, embed_dim)
        other_patches = patches[:, 1:, :]  # (batch_size, num_patches-1, embed_dim)

        # Create sequence: [CLS] + [prompts] + [other patches]
        patches_with_prompts = torch.cat([cls_patch, prompts, other_patches], dim=1)

        # Forward through frozen transformer encoder
        encoded_patches = self.backbone.forward_encoder(patches_with_prompts)

        # Extract features from prompt positions for classification
        selected_prompt_length = self.prompt_pool.top_k * self.prompt_pool.prompt_length
        features = encoded_patches[:, 1 : 1 + selected_prompt_length].mean(dim=1)
        # Final classification
        return self.head(features)  # (batch_size, num_classes)


class L2PPlugin(SupervisedPlugin):
    def before_backward(self, strategy: Any, *args, **kwargs) -> Any:
        strategy.loss += strategy.model.loss_

    def after_backward(self, strategy: Any, *args, **kwargs) -> Any:
        assert isinstance(strategy.model, L2PModel)
        assert strategy.model.loss_ is not None
        assert strategy.model.prompt_pool.keys.grad is not None
        assert strategy.model.prompt_pool.prompts.grad is not None
