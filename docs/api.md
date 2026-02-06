
## API文档

### models.DeepLearningClassifier

```python
from models.deep_learning import DeepLearningClassifier
model = DeepLearningClassifier(41, [256, 128, 64], 5, 0.3, "ReLU", True)
logits = model(torch.randn(8, 41))
```

- Args:
  - input_dim (int): 输入特征维度
  - hidden_dims (List[int]): 隐层维度列表
  - output_dim (int): 输出类别数
  - dropout (float): Dropout比例
  - activation (str): 激活函数名称
  - batch_norm (bool): 是否启用批归一化

- Methods:
  - forward(x: torch.Tensor) -> torch.Tensor
  - evaluate(data_loader, loss_fn, device) -> Dict[str, float]

### models.FeatureExtractor

```python
from models.feature_extractor import FeatureExtractor
extractor = FeatureExtractor(41)
features = extractor.extract(raw_sample)
```

- Args:
  - feature_dim (int): 输出特征维度

- Methods:
  - extract(raw_sample: Any) -> np.ndarray

### training.train

```python
from training.train import train_epoch, validate_epoch, train_loop
```

- train_epoch(model, data_loader, optimizer, loss_fn, device) -> Dict[str, float]
- validate_epoch(model, data_loader, loss_fn, device) -> Dict[str, float]
- train_loop(model, train_loader, val_loader, optimizer, loss_fn, max_epochs, patience, device, save_path, val_interval) -> Dict[str, list]

### inference.Predictor

```python
from inference.predictor import Predictor
predictor = Predictor(model, confidence_threshold=0.7)
result = predictor.predict_batch(torch.randn(8, 41))
pred_idx, conf = predictor.predict_one(torch.randn(41))
```

- Args:
  - model (nn.Module): 训练好的模型实例
  - device (torch.device): 推理设备
  - confidence_threshold (float): 置信度阈值

- Methods:
  - predict_batch(features: torch.Tensor) -> Dict[str, torch.Tensor]
  - predict_one(feature: torch.Tensor) -> Tuple[int, float]

## 常见问题

- Q: 为什么推理结果置信度低？
  - A: 检查特征归一化与模型训练是否充分，适当调整阈值。
- Q: 如何避免过拟合？
  - A: 使用早停、Dropout与批归一化，增大训练数据量。
- Q: GPU可用吗？
  - A: 将device设置为torch.device("cuda")并确保安装GPU版本PyTorch。
