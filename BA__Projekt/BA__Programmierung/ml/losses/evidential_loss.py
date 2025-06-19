# BA__Projekt/BA__Programmierung/ml/losses/evidential_loss.py

import torch


def evidential_loss(
    y: torch.Tensor,
    mu: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    lambda_reg: float = 1.0,
    kl_coef: float = 1e-2,
    use_logv: bool = False,
    mode: str = "full",  # ['nll', 'abs', 'mse', 'kl', 'full', 'scaled', 'variational']
) -> torch.Tensor:
    """
    Generalized Evidential Regression Loss supporting multiple variants.

    Parameters
    ----------
    y : torch.Tensor
        Ground truth targets.
    mu : torch.Tensor
        Predicted mean.
    v : torch.Tensor
        Predicted variance-related parameter (must be positive).
    alpha : torch.Tensor
        Evidence shape parameter.
    beta : torch.Tensor
        Evidence scale parameter.
    lambda_reg : float
        Coefficient for the regularization term.
    kl_coef : float
        Coefficient for the KL divergence term.
    use_logv : bool
        Use log1p(v) instead of v in regularization (for numerical stability).
    mode : str
        Loss variant: 'nll', 'abs', 'mse', 'kl', 'full', 'scaled', 'variational'.

    Returns
    -------
    torch.Tensor
        Scalar loss (mean over batch).
    """
    """
    Generalized Evidential Regression Loss with NaN protection and stability checks.
    """
    # ─── Sanitize Inputs ─── #
    v = v.clamp(min=1e-6)
    alpha = alpha.clamp(min=1.01)
    beta = beta.clamp(min=1e-6)

    if any(t.isnan().any() for t in (mu, v, alpha, beta)):
        print("❌ NaNs in inputs to evidential_loss")
        print(f"  mu:     {mu.mean().item():.4f},  v: {v.min().item():.2e} → {v.max().item():.2e}")
        print(f"  alpha:  {alpha.min().item():.2f} → {alpha.max().item():.2f}")
        print(f"  beta:   {beta.min().item():.2e} → {beta.max().item():.2e}")
        return torch.tensor(float("nan"), device=mu.device)

    try:
        two_blambda = 2 * beta * (1 + v)
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(two_blambda)
            + (alpha + 0.5) * torch.log((y - mu) ** 2 * v + two_blambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        if mode == "nll":
            return nll.mean()

        elif mode == "abs":
            error = torch.abs(y - mu)
            penalty = 2 * torch.log1p(v) if use_logv else 2 * v
            reg = error * (penalty + alpha)
            return (nll + lambda_reg * reg).mean()

        elif mode == "mse":
            error = (y - mu) ** 2
            penalty = 2 * torch.log1p(v) if use_logv else 2 * v
            reg = error * (penalty + alpha)
            return (nll + lambda_reg * reg).mean()

        elif mode == "kl":
            kl = (
                alpha * torch.log(beta)
                - (alpha - 0.5) * torch.digamma(alpha)
                + torch.lgamma(alpha)
                - 0.5 * torch.log(v)
                - alpha
            )
            return (nll + kl_coef * kl).mean()

        elif mode == "scaled":
            reg = torch.abs(y - mu) / (alpha + 1e-6)
            return (nll + lambda_reg * reg).mean()

        elif mode == "variational":
            precision = alpha / (beta + 1e-6)
            error = (y - mu) ** 2
            return 0.5 * (torch.log(1.0 / precision) + precision * error).mean()

        elif mode == "full":
            error = torch.abs(y - mu)
            penalty = 2 * torch.log1p(v) if use_logv else 2 * v
            reg = error * (penalty + alpha)
            kl = (
                alpha * torch.log(beta)
                - (alpha - 0.5) * torch.digamma(alpha)
                + torch.lgamma(alpha)
                - 0.5 * torch.log(v)
                - alpha
            )
            return (nll + lambda_reg * reg + kl_coef * kl).mean()

        else:
            raise ValueError(f"Unknown loss mode: '{mode}'")

    except Exception as e:
        print(f"Exception during evidential_loss({mode}): {e}")
        return torch.tensor(float("nan"), device=mu.device)


if __name__ == "__main__":
    torch.manual_seed(42)
    N = 16
    y = torch.randn(N)
    mu = y + 0.1 * torch.randn(N)
    v = torch.abs(torch.randn(N)) + 1e-6
    alpha = torch.abs(torch.randn(N)) + 1.0
    beta = torch.abs(torch.randn(N)) + 1.0

    print("=== Evidential Loss Variants ===\n")
    print(f"1) nll:         {evidential_loss(y, mu, v, alpha, beta, mode='nll'):.6f}")
    print(f"2) abs:         {evidential_loss(y, mu, v, alpha, beta, mode='abs'):.6f}")
    print(f"3) mse:         {evidential_loss(y, mu, v, alpha, beta, mode='mse'):.6f}")
    print(f"4) kl:          {evidential_loss(y, mu, v, alpha, beta, mode='kl'):.6f}")
    print(f"5) scaled:      {evidential_loss(y, mu, v, alpha, beta, mode='scaled'):.6f}")
    print(f"6) variational: {evidential_loss(y, mu, v, alpha, beta, mode='variational'):.6f}")
    print(f"7) full:        {evidential_loss(y, mu, v, alpha, beta, mode='full'):.6f}")

