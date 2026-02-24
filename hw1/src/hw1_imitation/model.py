"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        out_dim = chunk_size * action_dim
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # raise NotImplementedError
        B = state.shape[0]

        pred = self.net(state)  # (B, chunk_size * action_dim)
        pred = pred.view(B, self.chunk_size, self.action_dim)

        loss = self.loss_fn(pred, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
        B = state.shape[0]
        pred = self.net(state)
        return pred.view(B, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.action_dim_flat = chunk_size * action_dim

        in_dim = state_dim + self.action_dim_flat + 1

        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, self.action_dim_flat))

        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # raise NotImplementedError
        B = state.shape[0]

        x1 = action_chunk.view(B, -1)              
        x0 = torch.randn_like(x1)                  
        t = torch.rand(B, 1, device=state.device)  

        xt = (1 - t) * x0 + t * x1                  
        v_target = x1 - x0                         

        inp = torch.cat([state, xt, t], dim=1)
        v_pred = self.net(inp)

        loss = self.loss_fn(v_pred, v_target)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
        B = state.shape[0]

        x = torch.randn(
            B,
            self.action_dim_flat,
            device=state.device,
        )

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B, 1), i / num_steps, device=state.device)
            inp = torch.cat([state, x, t], dim=1)
            v = self.net(inp)
            x = x + dt * v

        return x.view(B, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
