from __future__ import annotations


def compute_confidence(score: float, threshold: float) -> float:
    """
    Compute a calibrated confidence value from the similarity score
    and operating threshold.

    Rule:
        confidence = 0.5 + 0.5 * ((score - threshold) / (1 - threshold))  if score >= threshold
        confidence = 0.5 * ((score - threshold) / (threshold + 1))        if score < threshold

    Interpretation:
        - Output range: [0.0, 1.0]
        - 0.5 means the score is exactly at the threshold (maximum uncertainty)
        - 1.0 means maximum confidence in a SAME decision (score = 1.0)
        - 0.0 means maximum confidence in a DIFFERENT decision (score = -1.0)
        - Values above 0.5 indicate SAME, below 0.5 indicate DIFFERENT

    This is a linear mapping that treats the threshold as the midpoint
    of the confidence scale, and scales symmetrically toward each extreme.
    """
    if score >= threshold:
        denom = max(1.0 - threshold, 1e-10)
        confidence = 0.5 + 0.5 * ((score - threshold) / denom)
    else:
        denom = max(threshold + 1.0, 1e-10)
        confidence = 0.5 + 0.5 * ((score - threshold) / denom)

    return float(max(0.0, min(1.0, confidence)))