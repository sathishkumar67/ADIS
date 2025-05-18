import torch
import torch.nn as nn

class EPU(nn.Module):
    r"""
    Exponential Partial Unit (EPU) activation:
    EPU(x) = clip(exp(k * x), theta_min, theta_max)

    Learnable parameters:
    - k: scaling factor for the exponent
    - theta_min: lower clipping bound
    - theta_max: upper clipping bound

    Initialization:
    - k ~ Uniform(0, 1)
    - theta_min ~ Uniform(-5, -1)
    - theta_max ~ Uniform(1, 5)
    """
    def __init__(self) -> None:
        super().__init__()
        # learnable parameters
        self.k = nn.Parameter(torch.empty(1))
        self.theta_min = nn.Parameter(torch.empty(1))
        self.theta_max = nn.Parameter(torch.empty(1))
        # initialize parameters
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        Initialize parameters:
        k ~ U(0,1), theta_min ~ U(-5,-1), theta_max ~ U(1,5)
        """
        nn.init.uniform_(self.k, a=0.0, b=1.0)
        nn.init.uniform_(self.theta_min, a=-5.0, b=-1.0)
        nn.init.uniform_(self.theta_max, a=1.0, b=5.0)
        # ensure valid ordering: theta_min < theta_max
        if self.theta_min >= self.theta_max:
            # swap if needed
            self.theta_min.data, self.theta_max.data = self.theta_max.data.clone(), self.theta_min.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_term = torch.exp(self.k * x)
        return torch.clamp(exp_term, self.theta_min, self.theta_max)