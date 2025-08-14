# Heston Surrogate Pricer Pipeline

## Tổng quan 
Project này triển khai phương pháp **Surrogate Pricer** cho calibration mô hình Heston theo tiếp cận hai giai đoạn:

1. **Giai đoạn R&D (Research & Development)**: Sử dụng Optuna để tìm siêu tham số tối ưu
2. **Giai đoạn Production**: Huấn luyện mô hình cuối cùng với best params và sử dụng cho calibration

## 🚀 Features

- **FFT-based Heston Pricing**: Semi-analytic option pricing using Carr-Madan FFT framework
- **Neural Network Calibration**: Advanced architectures (Attention, Residual blocks)  
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Two-Stage Approach**: R&D phase (Optuna) + Production phase (Final training)
- **GPU Acceleration**: Mixed-precision training for RTX 2060S
- **Professional Workflow**: Cookiecutter Data Science structure

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         heston_calib_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── heston_calib_project   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes heston_calib_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

