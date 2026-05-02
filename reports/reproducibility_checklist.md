# Reproducibility Checklist
**System:** LFW Face Verification  
**Final Tag:** `v1.0-final`  
**Milestone:** 4  

---

## Step 1: Clone and Setup

```powershell
# Clone the repository
git clone https://github.com/benanstrom/Faceoff_MSAI605.git
cd Faceoff_MSAI605

# Create and activate virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install facenet-pytorch --no-deps
pip install requests tqdm
pip install -e . --no-deps
```

---

## Step 2: Run Tests

```powershell
# Full test suite (16 tests, all should pass)
python -m pytest -q

# Milestone 3 inference tests only
python -m pytest tests/test_inference.py tests/test_smoke.py -v
```

**Expected:** 16 passed

---

## Step 3: Run Sample CLI Inference

```powershell
python scripts/infer_pairs.py \
  --pairs-csv artifacts/real_eval/m3_sample_pairs.csv \
  --output-json reports/infer_results.json
```

**Expected output format:**
```
Loading embedder: facenet
Running inference on 5 pairs...
[1/5] ------------------------------------------------------------
Image A    : lfw-deepfunneled/Jane_Fonda/Jane_Fonda_0002.jpg
Image B    : lfw-deepfunneled/Jane_Fonda/Jane_Fonda_0001.jpg
Score      : 0.076105
Threshold  : 0.6
Decision   : DIFFERENT
Confidence : 0.3363
Latency    : 0.2238s
------------------------------------------------------------
[3/5] ------------------------------------------------------------
Image A    : lfw-deepfunneled/George_Tenet/George_Tenet_0001.jpg
Image B    : lfw-deepfunneled/George_Tenet/George_Tenet_0002.jpg
Score      : 0.670348
Threshold  : 0.6
Decision   : SAME
Confidence : 0.5879
Latency    : 0.1126s
------------------------------------------------------------
```



**Output artifact:** `reports/infer_results.json`

---

## Step 4: Run Load Test

```powershell
python scripts/load_test.py \
  --pairs-csv artifacts/real_eval/m3_sample_pairs.csv \
  --num-workers 4 \
  --num-requests 20 \
  --output-json reports/load_test_results.json
```

**Expected:** 20/20 successful, throughput ~3-7 rps  
**Output artifact:** `reports/load_test_results.json`

---

## Step 5: Run Profiling

```powershell
python scripts/profile_inference.py \
  --pairs-csv artifacts/real_eval/m3_sample_pairs.csv \
  --output-json reports/profiling_results.json
```

**Expected stage breakdown (CPU baseline):**
- Preprocessing: ~7ms (3.76%)
- Embedding (x2): ~89ms (96.18%)
- Scoring: <1ms (0.06%)
- End-to-end: ~186ms

**Output artifact:** `reports/profiling_results.json`

---

## Step 6: Docker

```powershell
# Build
docker build -t faceoff-m3 .

# Run CLI inside container
docker run --rm faceoff-m3 python scripts/infer_pairs.py --help
```

---

## Key Artifact Paths

| Artifact | Path |
|---|---|
| System Card | `reports/system_card.md` |
| Profiling results | `reports/profiling_results.json` |
| Load test results | `reports/load_test_results.json` |
| Sample inference output | `reports/infer_results.json` |
| Inference config | `configs/m3_inference.yaml` |
| Sample pairs | `artifacts/real_eval/m3_sample_pairs.csv` |
| Final tag | `v1.0-final` |

---

## Key Config Values

| Config | Value |
|---|---|
| Embedding model | FaceNet InceptionResnetV1 (VGGFace2) |
| Operating threshold | 0.77 |
| Threshold selection rule | Maximize balanced accuracy on validation split |
| Image size | 160×160 |
| Similarity metric | Cosine similarity |