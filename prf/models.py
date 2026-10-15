"""Policies act in Euclidean space; retrieval normalizes the resulting action.

Log probabilities and KL are always calculated BEFORE normalization. They describe
latent actions, not densities on the sphere or probabilities of document relevance.
"""
import math
import torch
from torch import nn
from torch.distributions import Normal, Categorical, Independent, MixtureSameFamily
from torch.nn import functional as F


class Policy(nn.Module):
    def __init__(self, dim, kind="gaussian", hidden=64, components=4, prior_std=0.15):
        super().__init__()
        if kind not in {"dense", "perturb", "deterministic", "multi", "gaussian", "mixture"}:
            raise ValueError(kind)
        self.kind, self.dim, self.prior_std = kind, dim, prior_std
        self.k = components if kind in {"multi", "mixture"} else 1
        self.stochastic = kind in {"gaussian", "mixture"}
        if kind not in {"dense", "perturb"}:
            self.body = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
            self.mean = nn.Linear(hidden, self.k * dim)
            nn.init.normal_(self.mean.weight, std=0.01)
            nn.init.normal_(self.mean.bias, std=0.02)
            if self.stochastic:
                self.scale = nn.Linear(hidden, self.k * dim)
                nn.init.zeros_(self.scale.weight)
                nn.init.constant_(self.scale.bias, math.log(math.expm1(prior_std)))
                self.weight = nn.Linear(hidden, self.k)
                nn.init.zeros_(self.weight.weight)
                nn.init.zeros_(self.weight.bias)

    def parameters_for(self, q):
        h = self.body(q)
        mu = q[:, None, :] + self.mean(h).view(-1, self.k, self.dim)
        if not self.stochastic:
            return mu, None, None
        std = F.softplus(self.scale(h).view_as(mu)).clamp(1e-4, 2.0)
        weights = self.weight(h).softmax(-1)
        return mu, std, weights

    def distribution(self, q):
        mu, std, weights = self.parameters_for(q)
        return MixtureSameFamily(Categorical(probs=weights), Independent(Normal(mu, std), 1))

    def locations(self, q, samples):
        if samples < 1:
            raise ValueError("samples must be positive")
        if self.kind == "dense":
            return q[:, None, :]
        if self.kind == "perturb":
            return q[:, None, :] + self.prior_std * torch.randn(len(q), samples, self.dim)
        if self.stochastic:
            return self.distribution(q).sample((samples,)).transpose(0, 1)
        mu, _, _ = self.parameters_for(q)
        # One-vector baseline never repeats redundant searches. Multi-vector baseline
        # uses up to K heads; larger budgets deepen each search instead.
        return mu[:, :min(samples, self.k)]

    def surrogate_points(self, q, samples):
        mu, std, weights = self.parameters_for(q)
        if not self.stochastic:
            return mu[:, :, None, :], None
        eps = torch.randn(len(q), self.k, samples, self.dim, device=q.device)
        return mu[:, :, None, :] + std[:, :, None, :] * eps, weights

    def regularizer(self, q):
        mu, std, weights = self.parameters_for(q)
        if not self.stochastic:
            return (mu - q[:, None, :]).square().sum(-1).mean()
        # KL of joint (component, action) to uniform component/query Gaussian prior.
        # This is an upper bound on marginal mixture KL, NOT exact mixture KL.
        var = self.prior_std ** 2
        normal_kl = 0.5 * ((std.square() + (mu-q[:, None, :]).square()) / var
                           - 1 + 2 * torch.log(self.prior_std / std)).sum(-1)
        categorical_kl = (weights * (weights.clamp_min(1e-12).log() + math.log(self.k))).sum(-1)
        return ((weights * normal_kl).sum(-1) + categorical_kl).mean()
