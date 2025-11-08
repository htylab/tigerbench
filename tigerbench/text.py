"""
tigerbench.text
---------------
Text-based evaluation metrics for natural language and radiology report analysis.

Includes:
- BLEU
- ROUGE
- CheXbert macro-F1 (text → model → 14-label classification)
- RadGraph F1 (text → graph entity/relation evaluation)
"""

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score


# ---------------------------------------------------------------
# BLEU / ROUGE (general text metrics)
# ---------------------------------------------------------------
def bleu(refs, hyps):
    """
    Compute BLEU score between reference and hypothesis texts.
    """
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return {"score": float(bleu.score)}


def rouge(refs, hyps):
    """
    Compute ROUGE-1 and ROUGE-L F1 scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h) for r, h in zip(refs, hyps)]
    mean_r1 = np.mean([s["rouge1"].fmeasure for s in scores])
    mean_rl = np.mean([s["rougeL"].fmeasure for s in scores])
    return {"rouge1": float(mean_r1), "rougeL": float(mean_rl)}


# ---------------------------------------------------------------
# CheXbert F1 — uses pretrained CheXbert model (text → label → F1)
# ---------------------------------------------------------------
def chexbert_f1(refs, hyps, model_path=None, batch_size=8, device="cuda"):
    """
    Compute CheXbert macro-F1 between two sets of radiology reports.

    Args:
        refs, hyps: list of strings (radiology reports)
        model_path: optional local path to CheXbert weights
        batch_size: batch size for model inference
        device: "cuda" or "cpu"

    Returns:
        {"macro_f1": float}
    """
    try:
        from chexbert import CheXbert
        import torch  # noqa: F401  # torch is required by CheXbert internals
    except ImportError:
        raise ImportError(
            "CheXbert not installed. Please install dependencies:\n"
            "    pip install chexbert torch torchvision scikit-learn"
        )

    # Initialize model
    model = CheXbert(model_path=model_path, device=device)

    # Model inference: output shape (N, 14), values in {-1, 0, 1}
    y_true = np.asarray(model.infer(refs, batch_size=batch_size))
    y_pred = np.asarray(model.infer(hyps, batch_size=batch_size))

    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 14:
        raise ValueError("Expected output shape (N, 14) with values in {-1,0,1}")

    # Compute macro-F1 ignoring -1 labels
    f1s = []
    for i in range(14):
        true, pred = y_true[:, i], y_pred[:, i]
        mask = (true != -1) & (pred != -1)
        if np.any(mask):
            f1s.append(
                f1_score(true[mask], pred[mask], average="binary", zero_division=0)
            )

    macro = float(np.mean(f1s)) if f1s else 0.0
    return {"macro_f1": macro}


# ---------------------------------------------------------------
# RadGraph F1 — uses official RadGraph model for entity/relation F1
# ---------------------------------------------------------------
def radgraph_f1(refs, hyps, model_path=None, device="cuda"):
    """
    Compute RadGraph macro-F1 between reference and generated reports.

    Args:
        refs, hyps: list of report strings
        model_path: optional local path to RadGraph weights
        device: "cuda" or "cpu"

    Returns:
        {"macro_f1": float}
    """
    try:
        from radgraph import F1RadGraphEvaluator
    except ImportError:
        raise ImportError(
            "RadGraph evaluator not installed.\n"
            "Please install via:\n"
            "    pip install radgraph"
        )

    evaluator = F1RadGraphEvaluator(model_path=model_path, device=device)
    score = evaluator.evaluate(refs, hyps)
    return {"macro_f1": float(score)}
