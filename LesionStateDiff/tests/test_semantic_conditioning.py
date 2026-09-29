import torch
from torch import nn
from types import SimpleNamespace

from lesionstatediff.class_semantic_encoder import SpatialClassSemanticEncoder
from lesionstatediff.semantic_conditioning import SemanticConditionedUNet


class TinyMidBlock(nn.Module):
    def forward(self, hidden, _temb):
        return hidden * 2.0


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mid_block = TinyMidBlock()

    def forward(self, sample, timestep):
        temb = torch.zeros(sample.shape[0], 1, device=sample.device)
        return SimpleNamespace(sample=self.mid_block(sample, temb))


def test_semantic_encoder_shapes_and_zero_projection():
    encoder = SpatialClassSemanticEncoder(embedding_dim=32, output_channels=512)
    segmentation = torch.tensor([[[[0, 1], [2, 4]]]], dtype=torch.float32)
    output = encoder(segmentation, spatial_size=(3, 3), output_dtype=torch.float32)
    assert output.shape == (1, 512, 3, 3)
    assert torch.count_nonzero(output).item() == 0
    assert encoder.projection_is_zero()


def test_semantic_encoder_rejects_fractional_labels():
    encoder = SpatialClassSemanticEncoder(output_channels=4)
    segmentation = torch.tensor([[[[0.5]]]], dtype=torch.float32)
    try:
        encoder(segmentation, spatial_size=(1, 1), output_dtype=torch.float32)
    except ValueError as error:
        assert "integer class indices" in str(error)
    else:
        raise AssertionError("fractional segmentation label was accepted")


def test_projection_can_learn_after_zero_initialization():
    encoder = SpatialClassSemanticEncoder(embedding_dim=4, output_channels=2)
    segmentation = torch.tensor([[[[2, 3], [4, 1]]]], dtype=torch.float32)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
    output = encoder(segmentation, spatial_size=(2, 2), output_dtype=torch.float32)
    loss = (output - torch.ones_like(output)).square().mean()
    loss.backward()
    optimizer.step()
    assert not encoder.projection_is_zero()
    assert isinstance(encoder.embedding, nn.Embedding)


def test_zero_initialized_wrapper_matches_parent_output():
    parent = TinyUNet()
    sample = torch.randn(2, 4, 8, 8)
    segmentation = torch.randint(0, 5, (2, 1, 8, 8), dtype=torch.long).float()
    expected = parent(sample, torch.tensor([2, 2])).sample
    wrapped = SemanticConditionedUNet(parent, embedding_dim=4, mid_channels=4)
    observed = wrapped(
        sample, torch.tensor([2, 2]), semantic_seg=segmentation
    ).sample
    assert torch.equal(observed, expected)

