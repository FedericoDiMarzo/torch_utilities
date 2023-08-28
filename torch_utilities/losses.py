from torch import Tensor

from torch_utilities.common import pack_complex, phase


# export list
__all__ = ["ccmse_loss", "kld_normal_loss"]


def ccmse_loss(c: float = 0.3) -> Tensor:
    """
    Complex Compressed Mean-Squared Error loss
    | TOWARDS EFFICIENT MODELS FOR REAL-TIME DEEP NOISE SUPPRESSION
    | Authors: S. Braun et. al.

    Parameters
    ----------
    y_hat : Tensor
        Prediction of shape (B, 2, T, F)
    y_true : Tensor
        Label of shape (B, 2, T, F)
    c : float
        Compression coefficient for the spectrum,
        betweem 0 and 1
    """

    def _loss(y_hat: Tensor, y_true: Tensor) -> Tensor:
        """
        y_hat : Tensor
            Prediction of shape (B, 2, T, F)
        y_true : Tensor
            Label of shape (B, 2, T, F)
        """
        y_hat, y_true = [pack_complex(y) for y in (y_hat, y_true)]

        _f = lambda x, y, z, w: (
            ((x.abs() ** c) * z - (y.abs() ** c) * w).abs() ** 2
        ).mean()
        l1 = _f(y_hat, y_true, 1, 1)
        l2 = _f(y_hat, y_true, phase(y_hat), phase(y_true))
        loss = l1 + l2
        return loss

    return ccmse_loss


def kld_normal_loss(mu: Tensor, log_var: Tensor) -> Tensor:
    """
    Kullback-Leibler divergence w.r.t a standard normal distribution.

    Parameters
    ----------
    mu : Tensor
        Mean of the other distribution shape (B, T, F)
    log_var : Tensor
        Natural logarithm of the variance of the other distribution
        shape (B, T, F)

    Returns
    -------
    Tensor
        KLD loss between a normal distribution with mu=0 and logvar=0
        and another normal distribution
    """
    loss = (-0.5 * (1 + log_var - mu**2 - log_var.exp())).mean()
    return loss
