"""Tests for focal loss implementation."""

import torch

from turep import FocalLoss, focal_loss


class TestFocalLoss:
    """Test cases for focal loss."""

    def test_focal_loss_shape(self):
        """Test that focal loss returns correct shape."""
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        loss = focal_loss(logits, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_focal_loss_module(self):
        """Test FocalLoss module."""
        criterion = FocalLoss(gamma=2.0, alpha=1.0)
        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        loss = criterion(logits, targets)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_focal_loss_reduction(self):
        """Test different reduction modes."""
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))

        loss_mean = focal_loss(logits, targets, reduction="mean")
        loss_sum = focal_loss(logits, targets, reduction="sum")
        loss_none = focal_loss(logits, targets, reduction="none")

        assert loss_none.shape == torch.Size([8])
        assert loss_mean.shape == torch.Size([])
        assert loss_sum.shape == torch.Size([])
