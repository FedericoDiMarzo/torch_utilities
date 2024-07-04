from torch import nn
import pytest
import torch

import torch_utilities as tu

# Local fixtures ===============================================================


@pytest.fixture(params=[False, True])
def keep_graph(request) -> bool:
    """Keep graph for the gradient computation."""
    return request.param


# ==============================================================================


class TestPyTorch:
    @torch.set_grad_enabled(True)
    def test_compute_gradients(self, keep_graph):
        x = torch.ones((3, 2), requires_grad=True)
        y = x**2
        grad = tu.compute_gradients(x, y, keep_graph=keep_graph)
        z = grad.sum()
        assert z.item() <= 2 * 6
        if keep_graph:
            z.backward()
        else:
            with pytest.raises(RuntimeError):
                z.backward()

    def test_get_submodules(self):
        model = nn.Sequential(
            nn.Identity(),
            nn.ReLU(),
            nn.Tanh(),
            nn.Sequential(
                nn.SELU(),
                nn.Sigmoid(),
            ),
        )
        modules = tu.get_submodules(model)
        modules_types = [type(m) for m in modules]
        expected = [nn.Identity, nn.ReLU, nn.Tanh, nn.SELU, nn.Sigmoid]
        assert modules_types == expected

    @torch.set_grad_enabled(True)
    def test_freeze_model(self):
        module = nn.Conv2d(2, 2, 1)
        assert all(p.requires_grad for p in module.parameters())
        tu.freeze_model(module)
        assert not all(p.requires_grad for p in module.parameters())
