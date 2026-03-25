import torch


class DummyGather:
    def __init__(self, gathered):
        self._gathered = gathered

    def all_gather(self, tensor):
        return self._gathered

    def all_gather_cat(self, tensor_to_gather):
        gathered = self.all_gather(tensor_to_gather)
        if gathered.dim() > tensor_to_gather.dim():
            return gathered.flatten(0, 1)
        return gathered


def test_all_gather_cat_single_device_returns_original_shape_and_values():
    tensor = torch.randn(4, 3, 5)
    helper = DummyGather(gathered=tensor)
    out = helper.all_gather_cat(tensor)
    assert out.shape == tensor.shape
    assert torch.allclose(out, tensor)


def test_all_gather_cat_multi_device_stacks_and_flattens_correctly():
    tensor = torch.randn(4, 3, 5)
    # simulate two ranks returning the per-rank tensor
    gathered = torch.stack([tensor, tensor + 1.0], dim=0)  # (2,4,3,5)
    helper = DummyGather(gathered=gathered)
    out = helper.all_gather_cat(tensor)
    # should flatten leading rank and batch dims: (2*4,3,5)
    assert out.shape == (8, 3, 5)
    assert torch.allclose(out[:4], gathered[0])
    assert torch.allclose(out[4:], gathered[1])
