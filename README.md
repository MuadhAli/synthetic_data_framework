# Synthetic Data Generation Framework for Privacy-Preserving AI

A complete, production-quality Python framework for generating realistic 
**synthetic patient data** using state-of-the-art generative models (CTGAN, TVAE, 
GaussianCopula), built-in differential privacy, and demographic fairness auditing.  
No real patient data is required to run the pipeline.

---

## Project Overview

| Feature | Details |
|---|---|
| **Generative Models** | CTGAN, TVAE, GaussianCopula (SDV 1.9) |
| **Privacy** | Laplace DP via diffprivlib + Membership Inference evaluation |
| **Fairness** | Disparate Impact auditing via Aequitas |
| **ML Utility** | Train-on-Synthetic / Test-on-Real (TSTR) vs baseline |
| **Data Source** | Synthea patients.csv (or auto-generated mock data) |
| **Output** | Synthetic CSV + 4 PNG charts + console summary |

---

## Folder Structure

```
synthetic_data_framework/
├── requirements.txt        # pinned dependencies
├── README.md
├── config.py               # all hyperparameters in one place
├── main.py                 # full pipeline (python main.py)
│
├── data/
│   ├── raw/                # place patients.csv here (optional)
│   └── synthetic/          # synthetic_patients.csv saved here
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── gan_trainer.py
│   ├── privacy_layer.py
│   ├── bias_auditor.py
│   └── quality_evaluator.py
│
├── notebooks/
│   └── demo.ipynb          # full walk-through demo
│
└── results/
    └── charts/             # all PNG outputs saved here
```

---

## Installation

**Requirements**  
Python 3.9 – 3.11 recommended.

```bash
# 1. Navigate to the project directory
cd synthetic_data_framework

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Note:** `aequitas 0.42.0` requires `pandas < 2.0`.  
> If you encounter a conflict, install it via:  
> `pip install aequitas==0.42.0 --no-deps && pip install pandas==1.5.3`

---

## How to Run

### Option A — Full Pipeline (recommended)

```bash
python main.py
```

This executes all 7 steps end-to-end and prints a final summary.

### Option B — With Real Synthea Data

1. Download [Synthea](https://github.com/synthetichealth/synthea) and generate data:
   ```bash
   java -jar synthea-with-dependencies.jar -p 2000
   ```
2. Copy `output/csv/patients.csv` → `data/raw/patients.csv`
3. Run `python main.py`

### Option C — Jupyter Notebook Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

### Option D — Streamlit Demo UI (for presentation)

```bash
streamlit run streamlit_app.py
```

This launches an interactive dashboard with:
- full pipeline run controls,
- KPI cards (quality, privacy, bias, utility),
- result tabs and chart previews,
- synthetic CSV download button.

---

## Expected Output

```
════════════════════════════════════════════════════════════
  Synthetic Data Generation Framework  |  MTech Project
════════════════════════════════════════════════════════════

[STEP 1] Loading Synthea data...
  Shape            : (2000, 12)
  ...
[STEP 2] Training generative models...
  Training CTGAN  (epochs=300, batch_size=500) ...
  Training TVAE   (epochs=300, batch_size=500) ...
  Training GaussianCopula ...
[STEP 3] Generating synthetic data...
  Generated 1000 synthetic patient records.
[STEP 4] Applying differential privacy (Laplace mechanism)...
  Privacy budget ε = 1.0
[STEP 5] Running quality evaluation...
[STEP 6] Running ML utility test (TSTR vs TRTR)...
[PRIVACY] Running membership inference attack...
[STEP 7] Running bias audit...

════════════════════════════════════════════════════════════
  FRAMEWORK COMPLETE — FINAL SUMMARY
════════════════════════════════════════════════════════════
  Overall Quality Score               ~82.00%
  Privacy Risk Score (lower=safer)     ~15.00
  Bias Reduction                       ~12.00%
  TSTR Accuracy                        ~78.00%
  TRTR Accuracy (baseline)             ~81.00%
════════════════════════════════════════════════════════════
```

---

## Results Table

| Metric | Expected Range | Notes |
|---|---|---|
| Overall Quality Score | 75 – 92 % | Higher = better fidelity |
| Column Shapes Score | 80 – 95 % | Individual distribution fit |
| Column Pair Trends | 65 – 85 % | Correlation preservation |
| Privacy Risk Score | 10 – 30 | Lower = more private |
| TSTR Accuracy | 70 – 82 % | Proves ML utility |
| TRTR Accuracy | 78 – 85 % | Oracle baseline |
| Bias Reduction | 5 – 25 % | Positive = improved fairness |

---

## Generated Charts

| Chart | Path | Description |
|---|---|---|
| Training Loss | `results/charts/training_loss.png` | CTGAN generator loss per epoch |
| Quality Scores | `results/charts/quality_scores.png` | Per-column fidelity bar chart |
| Bias Comparison | `results/charts/bias_comparison.png` | DI ratio: real vs synthetic |
| Distribution | `results/charts/distribution_comparison.png` | AGE / GENDER / RACE overlay |

---

## Configuration

All hyperparameters live in `config.py`.  Key settings:

| Parameter | Default | Description |
|---|---|---|
| `CTGAN_EPOCHS` | 300 | CTGAN training epochs |
| `CTGAN_BATCH_SIZE` | 500 | Batch size |
| `N_SYNTHETIC_SAMPLES` | 1000 | Rows to generate |
| `DP_EPSILON` | 1.0 | Differential privacy budget |
| `TRAIN_TEST_SPLIT` | 0.80 | 80/20 train-test split |
| `RF_N_ESTIMATORS` | 100 | RandomForest estimators |

---

## Academic References

1. **CTGAN** — Xu et al., 2019. *Modeling Tabular Data using Conditional GAN.*  
2. **Differential Privacy** — Dwork, C., 2006. *Differential Privacy.*  
3. **Aequitas** — Saleiro et al., 2018. *Aequitas: A Bias and Fairness Audit Toolkit.*  
4. **Synthea** — Walonoski et al., 2018. *Synthea: An approach, method, and software mechanism for generating synthetic patients.*

---

## License

MIT License — for academic and research use.
