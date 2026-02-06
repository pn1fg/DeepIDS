import torch
from torch.utils.data import DataLoader, TensorDataset

from models.deep_learning import DeepLearningClassifier
from training.callbacks import EarlyStopping, ModelCheckpoint
from training.train import train, train_epoch, validate_epoch


def _make_loaders(batch_size: int = 32):
    features = torch.randn(128, 41)
    num_classes = DeepLearningClassifier().output_dim
    labels = torch.randint(0, num_classes, (128,))
    dataset = TensorDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size)
    return train_loader, val_loader


def test_train_epoch_and_validate_epoch():
    train_loader, val_loader = _make_loaders()
    model = DeepLearningClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metrics = train_epoch(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )
    assert train_metrics["loss"] >= 0.0
    assert 0.0 <= train_metrics["accuracy"] <= 1.0

    val_metrics = validate_epoch(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )
    assert val_metrics["loss"] >= 0.0
    assert 0.0 <= val_metrics["accuracy"] <= 1.0


def test_train_loop_saves_checkpoint(tmp_path):
    train_loader, val_loader = _make_loaders()
    model = DeepLearningClassifier()
    save_path = tmp_path / "best.pth"

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=2,
        patience=1,
        save_path=str(save_path),
        val_interval=1,
    )

    assert save_path.exists()
    assert len(history["train_loss"]) >= 1
    assert len(history["val_loss"]) >= 1


def test_early_stopping_behavior():
    early_stopping = EarlyStopping(patience=2, mode="max")
    assert early_stopping.step(0.5) is False
    assert early_stopping.step(0.49) is False
    assert early_stopping.step(0.48) is True


def test_model_checkpoint_behavior(tmp_path):
    model = DeepLearningClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = ModelCheckpoint(str(tmp_path / "best.pth"), monitor="val_accuracy")
    saved = checkpoint.step(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={"val_accuracy": 0.5},
    )
    assert saved is True
