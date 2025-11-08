from tigerbench import segmentation, reconstruction, classification, text

REGISTRY = {
    "segmentation": {
        "dice": segmentation.dice_multilabel,
        "iou": segmentation.iou_multilabel,
        "hd95": segmentation.hd95,
        "asd": segmentation.asd,
    },
    "reconstruction": {
        "mae": reconstruction.mae,
        "mse": reconstruction.mse,
        "psnr": reconstruction.psnr,
        "ssim": reconstruction.ssim_nd,
        "ncc": reconstruction.ncc,
        "mi": reconstruction.histogram_mi,
        "ksg_mi": reconstruction.ksg_mi,
    },
    "classification": {
        "accuracy": classification.accuracy,
        "precision": classification.precision,
        "recall": classification.recall,
        "f1": classification.f1,
    },
    "text": {
        "bleu": text.bleu,
        "rouge": text.rouge,
        "chexbert_f1": text.chexbert_f1,
        "radgraph_f1": text.radgraph_f1,
    },
}

# Flattened lookup for direct metric access
METRICS = {}
for category in REGISTRY.values():
    METRICS.update(category)


def get_metric(name):
    if name not in METRICS:
        available = sorted(METRICS.keys())
        raise ValueError(f"Unknown metric: {name}. Available: {available}")
    return METRICS[name]
