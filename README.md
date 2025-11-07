# tigerbench v0.0.1

輕量、nD、無 group 模式，支援 **個別參數設定**。

## 安裝
可以直接使用 pip 從 GitHub 取得最新版本，所有模組與相依套件預設一併安裝：

```bash
pip install https://github.com/htylab/tigerbench/archive/main.zip
```

若要安裝已發佈的穩定版本，可改用 PyPI：

```bash
pip install tigerbench==0.0.1
```

## 使用範例

```python
from tigerbench import evaluate
import numpy as np

# Dice: 只算 label 1,2
evaluate(gt, pred, "dice", labels=[1, 2])

# Dice + IoU: 只算 label 1
evaluate(gt, pred, ["dice", "iou"], labels=[1])

# KSG MI with k=5
evaluate(ref, img, "ksg_mi", k=5)

# Histogram MI with 128 bins
evaluate(ref, img, "mi", bins=128)

# F1 with micro average
evaluate(y_true, y_pred, "f1", average="micro")

# CheXbert F1
evaluate(labels_true, labels_pred, "chexbert_f1")
```

## 可用指標與參數

| 指標 | 可設參數 |
|------|----------|
| `dice`, `iou` | `labels`, `ignore_background` |
| `hd95`, `asd` | `labels` |
| `mi` | `bins` |
| `ksg_mi` | `k` |
| `precision`, `recall`, `f1` | `average` |
