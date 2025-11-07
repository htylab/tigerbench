import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score


def bleu(refs, hyps):
    if isinstance(refs[0], list):
        bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
    else:
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return {"score": float(bleu.score)}


def rouge(refs, hyps):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h) for r, h in zip(refs, hyps)]
    r1 = np.mean([s["rouge1"].fmeasure for s in scores]) if scores else 0.0
    rl = np.mean([s["rougeL"].fmeasure for s in scores]) if scores else 0.0
    return {"rouge1": float(r1), "rougeL": float(rl)}


CHEXBERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]


def chexbert_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 14:
        raise ValueError("Expected (N, 14) with values in {-1,0,1}")
    f1s = []
    for i in range(14):
        true, pred = y_true[:, i], y_pred[:, i]
        mask = (true != -1) & (pred != -1)
        if np.any(mask):
            f1s.append(f1_score(true[mask], pred[mask], average='binary', zero_division=0))
    macro = float(np.mean(f1s)) if f1s else 0.0
    return {"macro_f1": macro}


def radgraph_f1(refs, hyps):
    return {"macro_f1": 0.0, "note": "use official RadGraphF1 metric"}
