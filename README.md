PoC FRC Forecasting and Compliance (80/10/10)

Overview
- Compact PyTorch implementation for point-of-collection (PoC) free residual chlorine (FRC) forecasting and ≥0.2 mg/L compliance classification.
- Uses four CSVs (Sudan 2013, Jordan 2014/2015, Rwanda 2015) with se1_* inputs and se4_frc target.
- 80/10/10 train/val/test split with StandardScaler fit on train only.
- Outputs metrics CSVs and figures, and optionally runs a small hyperparameter grid.

Dataset
Place the following CSV files in a folder (e.g., `dataset/data/`):
- sudan_2013.csv
- jordan_2014.csv
- jordan_2015.csv
- rwanda_2015.csv

Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run
```
python implementation.py --data-dir dataset/data --out-dir .
```
This writes:
- classification_metrics_80_10_10.csv, regression_metrics_80_10_10.csv
- confusion_matrix.png, regression_scatter.png, ci_reliability.png

