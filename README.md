# 5G KPI Prediction and Anomaly Detection (4-week condensed project)

**Current stage:** condensed Week 2 (original weeks 3–4), covering 5G KPI preprocessing and traditional ML one-step forecasting. Future deep learning/anomaly detection/UI components are **not yet implemented in this repository**.

## Dataset and tasks

This stage uses the public **gotsf-ds** dataset (Hugging Face dataset ID `netop/gotsf-ds`) and four original columns: `DLPRB`, `DLThpVol`, `DLThpTime`, `MR_number`. Data origin: [dataset](https://huggingface.co/datasets/netop/gotsf-ds). Source script downloads four `*_train_0w-5w.csv` files.

- Each source CSV: 840 time steps and 2880 beams; aggregate long-format panel: **2,419,200 rows**.
- 30 base stations, 90 cells, 2880 beams (per recorded analysis report).
- Original-time split: train 0–671; held-out test 672–839.
- Target: each KPI at `t+1`; features: scaled KPI values for `t, t-1, t-2, t-24, t-168` within the same beam (20 features).
- Model-ready samples: 1,932,480 rows after lag/label boundaries; train 1,448,640 and test 483,840 (split by **target timestamp**).
- Methods: Persistence, Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM.
- All reported MAE/RMSE/R² are calculated at the **individual beam × target-time** level in original KPI scale. Prediction plots show **per-time beam means**, not the scoring granularity.

The source filenames carry the `train` designation; the last segment of those files is held out chronologically **within** those files. It is **not** an independent official test split.

## Files

```text
src/download_data.py           # Download four KPI source CSV files
src/audit_raw_data.py          # Validate and summarize raw inputs
src/preprocess.py              # Beam-level panel + train-fitted scaling
src/build_ml_features.py       # Group-wise lag features; t+1 targets
src/train_ml_models.py         # Six-model experiments + plots
reports/raw_data_audit.json
reports/raw_data_audit_summary.csv
reports/preprocessing_summary.json
reports/ml_feature_summary.json
reports/model_run_config.json
reports/model_metrics.csv
reports/prediction_timeseries.csv
reports/week3_data_analysis.md
reports/week4_model_evaluation.md
reports/week2_progress_report.md
figures/*_predictions.png
figures/model_comparison_*.png
```

Large CSV data and the `ml_features.pkl` cache are intentionally omitted from Git. The user's **local delivery ZIP** should include raw/processed data as required; absence from Git does not imply they were never generated.

## Reproduction

Tested original machine/package versions have **not** been captured in Git. `requirements.txt` lists supported version targets as an installation starting point; it is **not a lockfile** or exact proof of the versions used for the recorded metrics. For strict reproducibility, save `python --version`, `pip freeze`, OS/hardware details and source data checksums on the actual training machine.

Python 3.10+ (recommended: an isolated virtual environment):

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python src/download_data.py
python src/audit_raw_data.py
python src/preprocess.py
python src/build_ml_features.py
python src/train_ml_models.py
```

For a **quick smoke test** of model training, `python src/train_ml_models.py --quick` uses reduced settings and must **not** replace the full-size reported metrics. Use `python src/preprocess.py --force` and `python src/build_ml_features.py --force` to rebuild caches when needed. Note that the training job is memory- and compute-intensive because it uses up to 1.45 million training rows ×20 features across 4 targets and 5 fitted models.

In regions where `huggingface.co` is not reachable, `src/download_data.py` sets `HF_ENDPOINT=https://hf-mirror.com` by default.

## Recorded results

The existing `reports/model_run_config.json` states `quick_mode=false`; `reports/model_metrics.csv` stores 24 model-by-KPI results from a prior full-size run.

| KPI | Best MAE (model) | Best RMSE (model) | Best R² |
|---|---:|---:|---:|
| DLPRB | 0.261817 (XGBoost) | 0.493705 (RandomForest) | 0.787133 |
| DLThpVol | 0.207615 (RandomForest) | 0.460162 (RandomForest) | 0.615323 |
| DLThpTime | 0.243576 (XGBoost) | 0.524467 (RandomForest) | 0.702391 |
| MR_number | 0.224767 (XGBoost) | 0.395647 (RandomForest) | 0.867182 |

See [W4 model evaluation](reports/week4_model_evaluation.md) for metric definitions, full results, validity limitations, and details; see [condensed week 2 report](reports/week2_progress_report.md).

## Submission checklist

- [x] Raw data audit, preprocessing and feature-building source code
- [x] Existing time-series data statistical analysis
- [x] ML models, recorded metrics, prediction figures
- [x] W4 evaluation and condensed W2 weekly report
- [ ] Confirm the local ZIP includes `data/processed/kpi_timeseries.csv` and input raw files or an accessible regeneration path
- [ ] Record local `pip freeze` / hardware details and run from a clean environment
- [ ] Validate `python src/train_ml_models.py` in the student's environment if strict rerun is required

## Notes

This repository currently supports KPI **prediction**. In spite of the overall project title, anomaly detection and end-to-end 5G simulation integration are planned for subsequent phases; the existing gotsf-ds KPI dataset is **not** proof that free5GC/UERANSIM packet capture was successfully performed in this repository.
