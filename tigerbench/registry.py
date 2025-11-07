from tigerbench import segmentation, reconstruction, classification, text

METRICS = {
    # segmentation
    "dice": segmentation.dice_multilabel,
    "iou": segmentation.iou_multilabel,
    "hd95": segmentation.hd95,
    "asd": segmentation.asd,

    # reconstruction
    "mae": reconstruction.mae,
    "mse": reconstruction.mse,
    "psnr": reconstruction.psnr,
    "ssim": reconstruction.ssim_nd,
    "ncc": reconstruction.ncc,
    "mi": reconstruction.histogram_mi,
    "ksg_mi": reconstruction.ksg_mi,

    # classification
    "accuracy": classification.accuracy,
    "precision": classification.precision,
    "recall": classification.recall,
    "f1": classification.f1,

    # medical NLP
    "bleu": text.bleu,
    "rouge": text.rouge,
    "chexbert_f1": text.chexbert_f1,
    "radgraph_f1": text.radgraph_f1,
}

def get_metric(name):
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {sorted(METRICS.keys())}")
    return METRICS[name]
