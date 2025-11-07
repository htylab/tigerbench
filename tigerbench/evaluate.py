from tigerbench.registry import get_metric

def evaluate(y_true, y_pred, metrics, **kwargs):
    """
    統一評估入口
    - metrics: str 或 list[str]，僅接受實際 metric 名稱
    - kwargs: 傳給所有指定的 metric（各 metric 自行選擇使用）
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    if not all(isinstance(m, str) for m in metrics):
        raise ValueError("metrics must be str or list[str] of metric names")

    out = {}
    for m in metrics:
        func = get_metric(m)
        out[m] = func(y_true, y_pred, **kwargs)
    return {"metrics": out}
