"""
Verification metrics calculation.
A pair is accepted when score >= tau
"""

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def random_pair_split(n_pairs, validation_split, seed):
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_pairs)
    n_validation = int(round(validation_split * n_pairs))
    return np.sort(order[:n_validation]), np.sort(order[n_validation:])

def identity_disjoint_split(pair_ids, validation_split, seed):
    """
    Split pairs so that identities in the validation set are disjoint from the test set.
    Returns: (validation_indices, test_indices, n_dropped).
    """
    if any(a is None or b is None for a, b in pair_ids):
        raise ValueError("Identity is missing for at least one pair.")
    identities = sorted({i for pair in pair_ids for i in pair})
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(identities))
    n_validation = int(round(validation_split * len(identities)))
    validation_identities = {identities[k] for k in order[:n_validation]}
    validation, test, dropped = [], [], 0
    for index, (a, b) in enumerate(pair_ids):
        a_validation, b_validation = a in validation_identities, b in validation_identities
        if a_validation or b_validation:
            validation.append(index)
        elif not a_validation and not b_validation:
            test.append(index)
        else:
            dropped += 1
    return np.array(validation, dtype=int), np.array(test, dtype=int), dropped

def confusion_counts(scores, labels, tau):
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    accept = scores >= tau
    tp = int(np.sum(accept & labels))
    fp = int(np.sum(accept & ~labels))
    fn = int(np.sum(~accept & labels))
    tn = int(np.sum(~accept & ~labels))
    return tp, fp, fn, tn

def summarize_at(scores, labels, tau):
    """
    Counts and rates at one threshold.
    """
    tp, fp, fn, tn = confusion_counts(scores, labels, tau)
    total = tp + fp + fn + tn
    far = fp / (fp + tn) if (fp + tn) else float("nan")
    frr = fn / (fn + tp) if (fn + tp) else float("nan")
    return {
        "tau": float(tau),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "FAR": far,
        "FRR": frr,
        "TAR": 1.0 - frr if frr == frr else float("nan"),
        "accuracy": (tp + tn) / total if total else float("nan"),
    }

def threshold_for_far(scores, labels, target_far):
    """
    Find the smallest threshold that achieves the target false acceptance rate (FAR).
    Returns: tau
    """
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    impostor = np.sort(scores[~labels])[::-1] # descending
    if impostor.size == 0:
        raise ValueError("No impostor pairs in the set were used to choose tau.")
    allowed = int(np.floor(target_far * impostor.size))
    if allowed >= impostor.size:
        return float(impostor[-1])
    # accept scores that are above the allowed impostor threshold
    return float(np.nextafter(impostor[allowed], np.inf))

def equal_error_rate(scores, labels):
    """
    Compute the Equal Error Rate (EER) and the corresponding threshold.
    """
    fpr, tpr, thresholds = roc_curve(np.asarray(labels).astype(int), scores)
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[k] + fnr[k]) / 2.0), float(thresholds[k])

def far_frr_curve(scores, labels, n_points=400):
    scores = np.asarray(scores)
    taus = np.linspace(scores.min(), scores.max(), n_points)
    far = []
    frr = []
    for tau in taus:
        s = summarize_at(scores, labels, tau)
        far.append(s["FAR"])
        frr.append(s["FRR"])
    return taus, np.array(far), np.array(frr)

def auc(scores, labels):
    return float(roc_auc_score(np.asarray(labels).astype(int), scores))