import torch
import pytest
from torch import nn

from models.deep_learning import (
    DeepLearningClassifier,
    build_model,
    load_checkpoint,
    load_model,
    save_model,
)


def _expected_param_count(
    input_dim: int, hidden_dims: list[int], output_dim: int, batch_norm: bool
) -> int:
    total = 0
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        total += in_dim * hidden_dim + hidden_dim
        if batch_norm:
            total += hidden_dim * 2
        in_dim = hidden_dim
    total += in_dim * output_dim + output_dim
    return total


def _assert_state_dict_close(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        a_val = a[key]
        b_val = b[key]
        if isinstance(a_val, torch.Tensor):
            assert torch.allclose(a_val, b_val)
        elif isinstance(a_val, dict):
            _assert_state_dict_close(a_val, b_val)
        elif isinstance(a_val, (list, tuple)):
            assert len(a_val) == len(b_val)
            for left, right in zip(a_val, b_val):
                if isinstance(left, torch.Tensor):
                    assert torch.allclose(left, right)
                elif isinstance(left, dict):
                    _assert_state_dict_close(left, right)
                else:
                    assert left == right
        else:
            assert a_val == b_val


def test_model_init_structure_and_params():
    input_dim = 41
    hidden_dims = [256, 128, 64]
    output_dim = DeepLearningClassifier().output_dim
    model = DeepLearningClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=0.3,
        activation="ReLU",
        batch_norm=True,
    )

    linear_layers = [m for m in model.network if isinstance(m, nn.Linear)]
    assert len(linear_layers) == len(hidden_dims) + 1
    assert linear_layers[0].in_features == input_dim
    assert linear_layers[-1].out_features == output_dim

    expected_params = _expected_param_count(input_dim, hidden_dims, output_dim, True)
    actual_params = sum(p.numel() for p in model.parameters())
    assert actual_params == expected_params


def test_model_device_cpu_and_gpu():
    model = DeepLearningClassifier()
    cpu_input = torch.randn(4, model.input_dim)
    cpu_output = model(cpu_input)
    assert cpu_output.device.type == "cpu"

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    model = DeepLearningClassifier().to(device)
    gpu_input = torch.randn(4, model.input_dim, device=device)
    gpu_output = model(gpu_input)
    assert gpu_output.device.type == "cuda"


def test_forward_output_and_gradients():
    model = DeepLearningClassifier()
    batch = torch.randn(8, model.input_dim)
    output = model(batch)
    assert output.shape == (8, model.output_dim)
    assert torch.isfinite(output).all()

    labels = torch.randint(0, model.output_dim, (8,))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, labels)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_save_and_load_model_and_optimizer(tmp_path):
    output_dim = DeepLearningClassifier().output_dim
    model = build_model(
        {"input_dim": 41, "hidden_dims": [64, 32], "output_dim": output_dim}
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = torch.randn(6, model.input_dim)
    labels = torch.randint(0, model.output_dim, (6,))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(model(batch), labels)
    loss.backward()
    optimizer.step()

    save_path = tmp_path / "model.pt"
    save_model(
        model=model,
        save_path=str(save_path),
        optimizer=optimizer,
        epoch=3,
        metrics={"loss": 0.12},
    )

    loaded_model = load_model(str(save_path), device="cpu")
    for param, loaded_param in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(param, loaded_param)

    new_model = build_model(
        {"input_dim": 41, "hidden_dims": [64, 32], "output_dim": output_dim}
    )
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    payload = load_checkpoint(
        checkpoint_path=str(save_path),
        model=new_model,
        optimizer=new_optimizer,
        device="cpu",
    )
    _assert_state_dict_close(optimizer.state_dict(), new_optimizer.state_dict())
    assert payload["epoch"] == 3
    assert payload["metrics"] == {"loss": 0.12}


def test_model_info():
    output_dim = DeepLearningClassifier().output_dim
    model = DeepLearningClassifier(
        input_dim=41,
        hidden_dims=[32, 16],
        output_dim=output_dim,
        dropout=0.2,
        activation="ReLU",
        batch_norm=True,
    )
    info = model.get_model_info()
    assert info["architecture"] == [41, 32, 16, output_dim]

    expected_params = _expected_param_count(41, [32, 16], output_dim, True)
    assert info["total_params"] == expected_params
    assert info["trainable_params"] == expected_params
