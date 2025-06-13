# BA__Projekt/BA__Programmierung/ml/losses/evidential_loss.py
import torch


def evidential_loss(
    y: torch.Tensor,
    mu: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    lambda_reg: float = 1.0,
    kl_coef: float = 0.0,
    use_logv: bool = False,
    use_kl: bool = False,
    reg_type: str = "abs",  # or 'none'
) -> torch.Tensor:
    """
    Generalized Evidential Regression Loss with optional regularization and KL divergence.

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
    lambda_reg : float, optional
        Coefficient for regularization term (default: 1.0).
    kl_coef : float, optional
        Coefficient for KL divergence term (default: 0.0 — off).
    use_logv : bool, optional
        If True, apply log1p to v inside the regularizer (for log-space modeling).
    use_kl : bool, optional
        If True, adds KL divergence between predicted and standard Normal-Inverse-Gamma (NIG).
    reg_type : str, optional
        Type of regularization: `"abs"` (|y - μ| * penalty) or `"none"`.

    Returns
    -------
    torch.Tensor
        Mean evidential loss over batch.
    """
    two_blambda = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(torch.pi / v)
        - alpha * torch.log(two_blambda)
        + (alpha + 0.5) * torch.log((y - mu) ** 2 * v + two_blambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    reg = 0.0
    if reg_type == "abs":
        error = torch.abs(y - mu)
        penalty = 2 * torch.log1p(v) if use_logv else 2 * v
        reg = error * (penalty + alpha)

    kl = 0.0
    if use_kl:
        kl = (
            alpha * torch.log(beta)
            - (alpha - 0.5) * torch.digamma(alpha)
            + torch.lgamma(alpha)
            - 0.5 * torch.log(v)
            - alpha
        )

    return (nll + lambda_reg * reg + kl_coef * kl).mean()


if __name__ == "__main__":
    torch.manual_seed(42)
    N = 8
    y = torch.randn(N)
    mu = y + 0.1 * torch.randn(N)
    v = torch.abs(torch.randn(N)) + 1e-6  # positive variance-like param
    alpha = torch.abs(torch.randn(N)) + 1.0
    beta = torch.abs(torch.randn(N)) + 1.0

    print("=== Evidential Loss Variants ===\n")

    # 1) NLL only
    loss1 = evidential_loss(y, mu, v, alpha, beta, lambda_reg=0.0, kl_coef=0.0, reg_type="none")
    print(f"1) NLL only: {loss1.item():.6f}")

    # 2) NLL + L1 regularization (linear space)
    loss2 = evidential_loss(y, mu, v, alpha, beta, lambda_reg=1.0, kl_coef=0.0, reg_type="abs", use_logv=False)
    print(f"2) NLL + L1 reg (linear v): {loss2.item():.6f}")

    # 3) NLL + L1 regularization (log space)
    loss3 = evidential_loss(y, mu, v, alpha, beta, lambda_reg=1.0, kl_coef=0.0, reg_type="abs", use_logv=True)
    print(f"3) NLL + L1 reg (log1p v): {loss3.item():.6f}")

    # 4) NLL + KL divergence (no reg)
    loss4 = evidential_loss(y, mu, v, alpha, beta, lambda_reg=0.0, kl_coef=1e-2, use_kl=True, reg_type="none")
    print(f"4) NLL + KL divergence: {loss4.item():.6f}")

    # 5) NLL + L1 reg + KL divergence + log space reg
    loss5 = evidential_loss(
        y, mu, v, alpha, beta,
        lambda_reg=1.0,
        kl_coef=1e-2,
        use_kl=True,
        use_logv=True,
        reg_type="abs"
    )
    print(f"5) NLL + L1 reg + KL + log1p v: {loss5.item():.6f}")
