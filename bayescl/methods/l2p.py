# https://github.com/ContinualAI/avalanche/blob/master/avalanche/models/prompt.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.training.plugins import SupervisedPlugin
from loguru import logger
from torch import Tensor


class L2PViT(ABC):
    """Abstract interface for vision transformer backbones used in L2P."""

    @abstractmethod
    def get_embedding_size(self) -> int:
        """Get the dimension of the patch embeddings.

        :return: The embedding dimension.
        """

    @abstractmethod
    def get_patch_embed(self, pixel_values: Tensor) -> Tensor:
        """Turn pixel values into patch embeddings.

        :param pixel_values: A tensor of shape (batch_size, channels, height, width)
        :return: A tensor of shape (batch_size, num_patches + 1, embed_dim)
        """

    @abstractmethod
    def forward_encoder(self, prompts: Tensor, patch_embed: Tensor) -> Tensor:
        """Encode the patch embeddings with the given prompts.

        :param prompts: A tensor of shape (batch_size, prompt_length, embed_dim)
        :param patch_embed: A tensor of shape (batch_size, num_patches + 1, embed_dim)
        :return: A tensor of shape (batch_size, num_patches + 1 + prompt_length, embed_dim)
        """

    @abstractmethod
    def forward_query(self, patch_embed: Tensor) -> Tensor:
        """Get the encoded query embedding from the patch embeddings.

        :param patch_embed: A tensor of shape (batch_size, num_patches + 1, embed_dim)
        :return: A tensor of shape (batch_size, embed_dim)
        """


def _prompt_lookup(
    query: Tensor, keys: Tensor, prompts: Tensor, top_k: int
) -> Tuple[Tensor, Tensor]:
    """Find prompts with keys closest to the query.

    :param query: Query of shape (batch_size, embedding_dimension)
    :param keys: Keys of shape (pool_size, embedding_dimension)
    :param prompts: Prompts of shape (pool_size, prompt_length, embedding_dimension)
    :param top_k: Number of prompts to select
    :return: Tuple containing:
        - Selected prompts of shape (batch_size, top_k * prompt_length, embedding_dimension)
        - Average cosine distance between selected keys and query.
    """
    batch_size, embedding_dimension = query.shape
    pool_size, prompt_length, _ = prompts.shape
    assert query.shape == (batch_size, embedding_dimension)
    assert keys.shape == (pool_size, embedding_dimension)
    assert prompts.shape == (pool_size, prompt_length, embedding_dimension)
    assert top_k <= pool_size

    cosine_distance = 1 - F.cosine_similarity(query.unsqueeze(1), keys, dim=-1)
    _, idx = cosine_distance.topk(top_k, dim=1, largest=False)

    selected = prompts[idx].view(batch_size, top_k * prompt_length, embedding_dimension)
    return selected, cosine_distance[:, idx].mean()


class PromptPool(nn.Module):
    def __init__(
        self,
        prompts_per_task: int,
        prompt_length: int,
        embed_dim: int,
        top_k: int,
        num_tasks: int,
    ) -> None:
        super().__init__()
        self.prompts_per_task = prompts_per_task
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.num_tasks = num_tasks
        pool_size = prompts_per_task * num_tasks

        self.keys = nn.Parameter(torch.empty(pool_size, embed_dim))
        self.prompts = nn.Parameter(torch.empty(pool_size, prompt_length, embed_dim))

        nn.init.uniform_(self.keys, -1, 1)
        nn.init.uniform_(self.prompts, -1, 1)

    def forward(
        self,
        query: Tensor,
        task_id: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        if task_id is not None:
            start = task_id * self.prompts_per_task
            end = (task_id + 1) * self.prompts_per_task
            return _prompt_lookup(
                query=query,
                keys=self.keys[start:end],
                prompts=self.prompts[start:end],
                top_k=self.top_k,
            )

        return _prompt_lookup(
            query=query,
            keys=self.keys,
            prompts=self.prompts,
            top_k=self.top_k,
        )


class L2PModel(nn.Module):
    def __init__(
        self,
        vit: L2PViT,
        prompt_pool: PromptPool,
        out_features: int,
        pull_constraint_coeff: float,
    ):
        super().__init__()
        if isinstance(vit, nn.Module):
            vit = vit.eval().requires_grad_(False)
        else:
            raise ValueError("vit must be an instance of nn.Module.")

        self.vit = vit
        self.prompt_pool = prompt_pool
        self.head = nn.Linear(vit.get_embedding_size(), out_features)
        self.loss = 0
        self.train_task = 0
        self.pull_constraint_coeff = pull_constraint_coeff

    def forward(self, x: Tensor) -> Tensor:
        # First forward pass to get query
        patch_embed = self.vit.get_patch_embed(x)
        query = self.vit.forward_query(patch_embed)
        task_id = self.train_task if self.training else None
        prompts, cosine_distance = self.prompt_pool.forward(query, task_id=task_id)
        self.loss = self.pull_constraint_coeff * cosine_distance.mean()

        # Second forward pass this time with prompts
        encoded_patches = self.vit.forward_encoder(prompts, patch_embed)
        prompt_out_len = self.prompt_pool.top_k * self.prompt_pool.prompt_length

        # Trim [CLS] token and average prompt outputs
        features = encoded_patches[:, 1 : 1 + prompt_out_len].mean(dim=1)
        return self.head(features)


class L2PPlugin(SupervisedPlugin):
    def before_backward(self, strategy: Any, *args, **kwargs) -> Any:
        strategy.loss += strategy.model.loss

    def after_backward(self, strategy: Any, *args, **kwargs) -> Any:
        assert isinstance(strategy.model, L2PModel)
        assert strategy.model.loss is not None
        assert strategy.model.prompt_pool.keys.grad is not None
        assert strategy.model.prompt_pool.prompts.grad is not None

    def before_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        logger.info(f"Training task {strategy.clock.train_exp_counter}")
        strategy.model.train_task = strategy.clock.train_exp_counter
