import torch
from torch.utils.data import TensorDataset

from models.deep_learning import DeepLearningClassifier
from training.hyperparameter_opt import (
    optimize_hyperparameters,
    save_best_params,
    visualize_optimization_history,
    visualize_parameter_importance,
    visualize_parameter_relationships,
)


def test_optimize_and_visualize(tmp_path):
    torch.manual_seed(0)
    features = torch.randn(80, 41)
    num_classes = DeepLearningClassifier().output_dim
    labels = torch.randint(0, num_classes, (80,))

    train_dataset = TensorDataset(features[:64], labels[:64])
    val_dataset = TensorDataset(features[64:], labels[64:])

    result = optimize_hyperparameters(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_trials=2,
        max_epochs=2,
        patience=1,
        device=torch.device("cpu"),
    )

    assert "best_params" in result
    assert "study" in result
    assert result["best_params"]

    study = result["study"]
    history_path = tmp_path / "history.png"
    importance_path = tmp_path / "importance.png"
    relationship_path = tmp_path / "relationship.png"
    params_path = tmp_path / "best_params.yaml"

    visualize_optimization_history(study, str(history_path))
    visualize_parameter_importance(study, str(importance_path))
    visualize_parameter_relationships(study, str(relationship_path))
    save_best_params(result["best_params"], str(params_path))

    assert history_path.exists()
    assert importance_path.exists()
    assert relationship_path.exists()
    assert params_path.exists()
