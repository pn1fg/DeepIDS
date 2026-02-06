"""
深度学习分类模型模块。

该模块定义可配置的全连接神经网络分类器，用于入侵检测任务。

使用示例:
    import torch
    from models.deep_learning import DeepLearningClassifier

    model = DeepLearningClassifier(
        input_dim=41,
        hidden_dims=[256, 128, 64],
        output_dim=22,
        dropout=0.3,
        activation="ReLU",
        batch_norm=True,
    )
    batch = torch.randn(8, 41)
    logits = model(batch)

错误处理说明:
    训练与评估过程中的异常应被捕获并转换为可读的运行时错误。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch import nn
from torch.nn.utils import prune

from utils.config import ATTACK_TYPES


class DeepLearningClassifier(nn.Module):
    """
    入侵检测深度学习分类器。

    Args:
        input_dim: 输入特征维度。
        hidden_dims: 隐层维度列表。
        output_dim: 输出类别数。
        dropout: Dropout比例。
        activation: 激活函数名称。
        batch_norm: 是否启用批归一化。

    使用示例:
        model = DeepLearningClassifier(41, [256, 128, 64], 22, 0.3, "ReLU", True)
    """

    def __init__(
        self,
        input_dim: int = 41,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = len(ATTACK_TYPES),
        dropout: float = 0.3,
        activation: str = "ReLU",
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim和output_dim必须为正整数")
        if not hidden_dims:
            raise ValueError("hidden_dims不能为空")
        if dropout < 0.0 or dropout > 0.9:
            raise ValueError("dropout应在0.0到0.9之间")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(self.activation))
            layers.append(nn.Dropout(self.dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        return nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        if name == "ReLU":
            return nn.ReLU()
        raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入特征张量，形状为(batch_size, input_dim)。

        Returns:
            logits: 未归一化的分类输出。
        """
        return self.network(x)

    def get_model_info(self) -> Dict[str, Union[int, List[int], Dict[str, int]]]:
        """
        获取模型架构与参数统计信息。

        Returns:
            包含架构、总参数量与可训练参数量的字典。
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        architecture = [self.input_dim, *self.hidden_dims, self.output_dim]
        return {
            "architecture": architecture,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
        }

    def train(self, mode: bool = True) -> "DeepLearningClassifier":
        """
        设置模型为训练或评估模式。

        Args:
            mode: True为训练模式，False为评估模式。

        Returns:
            self
        """
        super().train(mode)
        return self

    def evaluate(
        self,
        data_loader: Iterable[torch.Tensor],
        loss_fn: nn.Module,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """
        在给定数据集上评估模型。

        Args:
            data_loader: 评估数据加载器。
            loss_fn: 损失函数。
            device: 计算设备。

        Returns:
            包含平均损失与准确率的字典。
        """
        device = device or torch.device("cpu")
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        try:
            with torch.no_grad():
                for batch in data_loader:
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    logits = self(features)
                    loss = loss_fn(logits, labels)
                    total_loss += float(loss.item()) * labels.size(0)
                    preds = torch.argmax(logits, dim=1)
                    total_correct += int((preds == labels).sum().item())
                    total_samples += int(labels.size(0))
        except Exception as exc:
            raise RuntimeError("评估过程发生异常") from exc

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return {"loss": avg_loss, "accuracy": accuracy}


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 41,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = len(ATTACK_TYPES),
        bidirectional: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim和output_dim必须为正整数")
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim和num_layers必须为正整数")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("输入维度不合法")
        _, (hn, _) = self.lstm(x)
        if self.bidirectional:
            last = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            last = hn[-1]
        return self.fc(last)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 41,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        output_dim: int = len(ATTACK_TYPES),
        dropout: float = 0.2,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim和output_dim必须为正整数")
        if model_dim <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("model_dim、num_layers与num_heads必须为正整数")
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.dropout = dropout
        self.max_len = max_len
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Embedding(max_len, model_dim)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("输入维度不合法")
        seq_len = x.size(1)
        pos_idx = torch.arange(seq_len, device=x.device) % self.max_len
        pos = self.pos_embed(pos_idx).unsqueeze(0)
        h = self.input_proj(x) + pos
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.fc(pooled)


class CNN1DClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 41,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        output_dim: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim和output_dim必须为正整数")
        if kernel_size <= 0:
            raise ValueError("kernel_size必须为正整数")
        self.input_dim = input_dim
        self.channels = channels or [32, 64]
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.dropout = dropout
        layers: List[nn.Module] = []
        in_ch = 1
        for ch in self.channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3:
            x = x.reshape(x.size(0), -1).unsqueeze(1)
        else:
            raise ValueError("输入维度不合法")
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


def build_model(config: Dict[str, Any], device: str = "cpu") -> nn.Module:
    """
    根据配置构建模型。

    Args:
        config: 包含模型参数的配置字典。
        device: 设备类型("cpu"或"cuda")。

    Returns:
        初始化好的模型。
    """
    model_cfg = config.get("model") if isinstance(config, dict) else None
    if model_cfg is None:
        model_cfg = config
    architecture = str(model_cfg.get("architecture", "mlp")).lower()
    input_dim = int(model_cfg.get("input_dim", 41))
    output_dim = int(model_cfg.get("output_dim", len(ATTACK_TYPES)))
    dropout = float(model_cfg.get("dropout", 0.3))
    if architecture in {"mlp", "fc", "dense"}:
        hidden_dims = list(model_cfg.get("hidden_dims", [256, 128, 64]))
        activation = str(model_cfg.get("activation", "ReLU"))
        batch_norm = bool(model_cfg.get("batch_norm", True))
        model = DeepLearningClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )
    elif architecture == "lstm":
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        num_layers = int(model_cfg.get("num_layers", 2))
        bidirectional = bool(model_cfg.get("bidirectional", False))
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            bidirectional=bidirectional,
            dropout=dropout,
        )
    elif architecture == "transformer":
        model_dim = int(model_cfg.get("model_dim", 64))
        num_layers = int(model_cfg.get("num_layers", 2))
        num_heads = int(model_cfg.get("num_heads", 4))
        max_len = int(model_cfg.get("max_len", 256))
        model = TransformerClassifier(
            input_dim=input_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=output_dim,
            dropout=dropout,
            max_len=max_len,
        )
    elif architecture == "cnn":
        channels = list(model_cfg.get("channels", [32, 64]))
        kernel_size = int(model_cfg.get("kernel_size", 3))
        model = CNN1DClassifier(
            input_dim=input_dim,
            channels=channels,
            kernel_size=kernel_size,
            output_dim=output_dim,
            dropout=dropout,
        )
    else:
        raise ValueError("不支持的architecture配置")
    model.to(torch.device(device))
    return model


def save_model(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    保存模型、优化器和训练信息。
    """
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    model_config: Dict[str, Any] = {
        "input_dim": int(getattr(model, "input_dim", 0)),
        "output_dim": int(getattr(model, "output_dim", 0)),
        "dropout": float(getattr(model, "dropout", 0.0)),
    }
    if isinstance(model, DeepLearningClassifier):
        model_config.update(
            {
                "architecture": "mlp",
                "hidden_dims": list(model.hidden_dims),
                "activation": model.activation,
                "batch_norm": model.batch_norm,
            }
        )
    elif isinstance(model, LSTMClassifier):
        model_config.update(
            {
                "architecture": "lstm",
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "bidirectional": model.bidirectional,
            }
        )
    elif isinstance(model, TransformerClassifier):
        model_config.update(
            {
                "architecture": "transformer",
                "model_dim": model.model_dim,
                "num_layers": model.num_layers,
                "num_heads": model.num_heads,
                "max_len": model.max_len,
            }
        )
    elif isinstance(model, CNN1DClassifier):
        model_config.update(
            {
                "architecture": "cnn",
                "channels": list(model.channels),
                "kernel_size": model.kernel_size,
            }
        )
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": model_config,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if metrics is not None:
        payload["metrics"] = dict(metrics)
    torch.save(payload, save_path)


def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    加载模型。
    """
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_cfg = checkpoint.get("model_config") or {}
        model = build_model(model_cfg, device=device)
        model.load_state_dict(checkpoint["model_state"])
    elif isinstance(checkpoint, dict):
        model = DeepLearningClassifier().to(torch.device(device))
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("模型文件格式不受支持")
    return model


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    加载检查点（用于恢复训练）。
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if not isinstance(checkpoint, dict):
        raise ValueError("检查点格式不受支持")
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    payload: Dict[str, Any] = {}
    if "epoch" in checkpoint:
        payload["epoch"] = checkpoint["epoch"]
    if "metrics" in checkpoint:
        payload["metrics"] = checkpoint["metrics"]
    return payload


def quantize_model(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )


def prune_model(model: nn.Module, amount: float = 0.2) -> nn.Module:
    if amount <= 0.0 or amount >= 1.0:
        raise ValueError("amount应在0到1之间")
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model
