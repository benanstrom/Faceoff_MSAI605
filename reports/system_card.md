# System Card: LFW Face Verification System
**Version:** v1.0-final  
**Milestone:** 4  
**Date:** May 2026  

---

## 1. System Overview

This system is a face verification pipeline built on the Labeled Faces in the Wild (LFW) dataset. Given two face images, the system produces a similarity score, a binary same-person or different-person decision, a calibrated confidence value, and a latency measurement.

**Pipeline:**
Image A + Image B
- Preprocess (resize 160×160, normalize [-1, 1])
- FaceNet InceptionResnetV1 (512-dim embeddings, pretrained VGGFace2)
- Cosine Similarity Score
- Threshold Decision (threshold: 0.77)
- Calibrated Confidence (linear scaling, range [0.0, 1.0])
- Output: score, decision, confidence, latency

**Key components:**
- Embedding model: FaceNet InceptionResnetV1 pretrained on VGGFace2
- Similarity metric: Cosine similarity
- Decision threshold: 0.77 (selected on validation split)
- Confidence: Linear threshold scaling — at threshold = 0.5, score 1.0 = 1.0, score -1.0 = 0.0

---

## 2. Intended Use

**Intended uses:**
- Academic research and coursework on face verification pipelines
- Benchmarking embedding-based verification on LFW
- Prototyping and evaluating face verification system components

**Out-of-scope uses:**
- Production identity verification or access control systems
- Law enforcement, surveillance, or security screening
- Any application requiring certified fairness or demographic parity guarantees
- Real-time high-throughput deployment without hardware-aware optimization
- Verification of faces outside the LFW distribution (e.g. non-frontal, heavily occluded, or low-resolution images)

---

## 3. Data Summary

- **Dataset:** Labeled Faces in the Wild (LFW), deep-funneled variant
- **Total identities:** 5,749
- **Total images:** 13,233
- **Pair construction:** Deterministic positive/negative pair generation with `prefer_unique` policy to reduce identity dominance
- **Splits:** Train / Validation / Test — threshold selected on validation, final metrics reported on test
- **Total evaluated pairs:** 10,000 (5,000 same-identity, 5,000 different-identity)

**Data limitations:**
- LFW is heavily skewed toward public figures, predominantly male, and predominantly Western
- Many identities have only one or two images, limiting diversity of same-identity pairs
- Deep-funneled alignment may not generalize to unaligned or non-frontal inputs
- No reliable demographic metadata is available for subgroup analysis

---

## 4. Operating Threshold and Key Metrics

**Threshold selection rule:** Maximize balanced accuracy on the validation split.

**Final system (improved, 32×32 baseline representation → embedding-based):**

| Metric | Value |
|---|---|
| Selected threshold | 0.77 |
| Validation balanced accuracy | 0.61 |
| Test balanced accuracy | 0.5886 |
| Test accuracy | 0.5886 |
| Precision | 0.587 |
| Recall | 0.597 |
| True Positives | 1,493 |
| False Positives | 1,050 |
| True Negatives | 1,450 |
| False Negatives | 1,007 |

**Score direction:** Higher cosine similarity → more likely same-person pair.

**Confidence interpretation:**
- 0.5 = score exactly at threshold (maximum uncertainty)
- Above 0.5 = SAME decision
- Below 0.5 = DIFFERENT decision
- 1.0 = maximum SAME confidence
- 0.0 = maximum DIFFERENT confidence

---

## 5. Failure Modes and Limitations

**Known failure modes:**

1. **Pose and lighting variation:** Same-identity pairs with significant pose change, strong shadows, or unusual lighting produce lower similarity scores and are more likely to be incorrectly classified as DIFFERENT.

2. **Visually similar different-identity pairs:** Pairs of different individuals who share similar facial structure, age, or ethnicity can produce high similarity scores and be incorrectly classified as SAME.

3. **Low-image-count identities:** Identities with only one or two images in LFW tend to produce harder same-identity pairs since the available images may vary significantly in capture conditions.

4. **Near-threshold scores:** Pairs with scores close to 0.77 are inherently ambiguous. Small changes in image quality or preprocessing can flip the decision.

5. **Image quality degradation:** Low-resolution, blurry, or heavily compressed inputs are not filtered before inference. The pipeline assumes clean, reasonably frontal face images at standard LFW quality.

6. **Occlusion:** Partially occluded faces (glasses, masks, hair) are not handled specially and will degrade embedding quality.

---

## 6. Fairness-Related Risks

This system has not been evaluated on demographic subgroups due to the absence of reliable demographic metadata in LFW.

**Known risk categories:**

- **Dataset demographic skew:** LFW is skewed toward public figures who are predominantly male and Western. Verification performance may be uneven across demographic groups not well represented in the dataset.

- **Embedding model bias:** FaceNet InceptionResnetV1 was pretrained on VGGFace2. Any demographic imbalance in VGGFace2 may carry over into the embedding representations used by this system.

- **Image quality disparities:** If image quality differs systematically across demographic groups, the system may perform unevenly without this being visible in aggregate metrics.

- **Misuse risk:** This system should not be used in any context where a false match or false non-match could harm an individual, including identity verification, access control, or surveillance.

**We do not make unsupported demographic performance claims.** Subgroup analysis would require verified demographic labels not present in this dataset.

---

## 7. Operational Constraints

| Constraint | Detail |
|---|---|
| Input format | JPEG or PNG face images, reasonably frontal, standard LFW quality |
| Input size | Resized internally to 160×160 pixels |
| Embedding model | FaceNet InceptionResnetV1, pretrained VGGFace2, ~107MB download on first run |
| Hardware | CPU-only baseline; no GPU required |
| Preprocessing latency | ~7ms per image |
| Embedding latency | ~89ms per image (CPU) |
| Scoring latency | <1ms per pair |
| End-to-end latency | ~186ms per pair (CPU) |
| Throughput | ~5–7 pairs/second (CPU, batch size 1–4) |
| Concurrency | Tested with 4 workers, 20 requests; 20/20 successful |
| OS | Tested on Windows 11; Docker image based on python:3.11-slim |
| Python | 3.11+ |
| Key dependencies | torch, facenet-pytorch, Pillow, numpy, PyYAML |

---

## 8. Reproducibility

- **Final tag:** `v1.0-final`
- **README:** See repository root `README.md` for full setup, Docker, CLI, and test commands
- **Reproducibility checklist:** `reports/reproducibility_checklist.md`
- **Profiling report:** `reports/profiling_results.json`
- **Inference config:** `configs/m3_inference.yaml`
- **Sample pairs:** `artifacts/real_eval/m3_sample_pairs.csv`
- **Sample inference output:** `reports/infer_results.json`
- **Load test results:** `reports/load_test_results.json`

A grader can reproduce the core CLI inference by following the reproducibility checklist from a fresh clone without access to the full LFW dataset, using the committed sample pairs.