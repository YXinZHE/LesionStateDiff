import torch

from lesionstatediff.class_semantic_encoder import SpatialClassSemanticEncoder


def test_trained_semantic_state_is_not_reinitialized_on_strict_reload():
    trained = SpatialClassSemanticEncoder(embedding_dim=4, output_channels=3)
    with torch.no_grad():
        trained.projection.weight.fill_(0.125)
        trained.projection.bias.fill_(-0.25)
    expected = {key: value.clone() for key, value in trained.state_dict().items()}

    restored = SpatialClassSemanticEncoder(embedding_dim=4, output_channels=3)
    result = restored.load_state_dict(expected, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert not restored.projection_is_zero()
    for key, value in expected.items():
        assert torch.equal(restored.state_dict()[key], value)


def test_refinement_optimizer_keeps_unet_and_semantic_groups_separate():
    unet_parameter = torch.nn.Parameter(torch.ones(1))
    semantic_parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(
        [
            {"params": [unet_parameter], "lr": 1e-7, "name": "unet"},
            {"params": [semantic_parameter], "lr": 1e-5, "name": "semantic_branch"},
        ]
    )
    assert optimizer.param_groups[0]["lr"] == 1e-7
    assert optimizer.param_groups[1]["lr"] == 1e-5

