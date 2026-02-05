"""
Sharpness-Aware Minimization (SAM) Optimizer

SAM seeks to find parameters that lie in flat minima of the loss landscape,
which has been shown to improve generalization. This is particularly
important for physics-informed neural networks where we want the model
to generalize to new airfoil configurations.

Reference:
    Foret et al., "Sharpness-Aware Minimization for Efficiently 
    Improving Generalization", ICLR 2021
"""

import torch
from torch.optim import Optimizer
from typing import Callable, Optional


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer.
    
    SAM simultaneously minimizes loss value and loss sharpness by:
    1. First step: Perturbing weights to climb to local loss peak
    2. Second step: Computing gradients at perturbed location
    3. Returning to original weights and applying update
    
    This results in parameters that lie in "flat" regions of the
    loss landscape, improving generalization.
    """
    
    def __init__(
        self, 
        params, 
        base_optimizer: type, 
        rho: float = 0.05, 
        adaptive: bool = False, 
        **kwargs
    ):
        """
        Initialize SAM optimizer.
        
        Args:
            params: Model parameters to optimize
            base_optimizer: Base optimizer class (e.g., torch.optim.Adam)
            rho: Neighborhood size for sharpness estimation (default: 0.05)
            adaptive: Whether to use adaptive rho (default: False)
            **kwargs: Additional arguments passed to base optimizer
        """
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        Perform the first step: perturb weights to local loss maximum.
        
        This step computes the perturbation direction (gradient direction)
        and moves the weights to estimate the sharpest point nearby.
        
        Args:
            zero_grad: Whether to zero gradients after step
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Store the perturbation for later removal
                self.state[p]["e_w"] = p.grad * scale.to(p)
                # Move to perturbed location
                p.add_(self.state[p]["e_w"])
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        Perform the second step: return to original weights and update.
        
        This step removes the perturbation and applies the standard
        optimizer update using gradients computed at the perturbed point.
        
        Args:
            zero_grad: Whether to zero gradients after step
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Return to original position
                p.sub_(self.state[p]["e_w"])
        
        # Apply base optimizer update
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self) -> torch.Tensor:
        """
        Compute the L2 norm of all gradients.
        
        Returns:
            L2 norm of concatenated gradients
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def step(self, closure: Optional[Callable] = None) -> None:
        """
        Standard step interface (not used in SAM, use first_step/second_step).
        """
        raise NotImplementedError(
            "SAM requires two-step optimization. "
            "Use first_step() and second_step() instead."
        )
