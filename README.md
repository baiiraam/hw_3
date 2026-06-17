# AI-ENG-201 – Homework 3
## Data Cleaning, Feature Engineering, and Data Visualization

**Author:** Bayram Bayramov
\
**Date:** 17.06.2026
\
**Course:** AI-ENG-201 – Machine Learning (Fall 2026), NAIC

---

## 📋 Project Overview

This assignment implements a complete data preprocessing and machine learning pipeline from scratch, covering:

- **Data Cleaning:** Missing value analysis (MCAR/MAR/MNAR), custom imputation (SimpleImputer, KNNImputer)
- **Feature Engineering:** Custom scalers (StandardScaler, MinMaxScaler, RobustScaler), polynomial features, binning
- **Categorical Encoding:** One-hot encoding, CV-safe target encoding
- **Visualization:** Anscombe's Quartet, Titanic EDA, Tufte's principles
- **Integration:** sklearn-compatible Pipeline with custom transformers
- **Bonus:** Yeo-Johnson transformation, comprehensive visualization report

All custom implementations are written from scratch without using sklearn's preprocessing modules (except for reference/comparison).

---

## 📁 Directory Structure

```
hw3/
├── src/                           # Custom implementations
│   ├── __init__.py
│   ├── imputers.py                # SimpleImputer, KNNImputer (with KD-Tree)
│   ├── encoders.py                # OneHotEncoder (sparse), TargetEncoder (CV-safe)
│   ├── scalers.py                 # StandardScaler, MinMaxScaler, RobustScaler
│   └── feature_creation.py        # PolynomialFeatures, KBinsDiscretizer
├── notebooks/
│   └── hw3_analysis.ipynb            # Complete analysis notebook
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py      # 14 tests (all passing)
├── figures/                       # All generated figures
│   ├── missingness_heatmap.png
│   ├── imputation_comparison_correct.png
│   ├── scaling_statistical_comparison.png
│   ├── scaling_qqplots.png
│   ├── scaling_overlay.png
│   ├── binning_comparison.png
│   ├── anscombe_quartet.png
│   ├── titanic_pairplot.png
│   ├── titanic_survival_by_class.png
│   ├── titanic_age_boxplot.png
│   ├── yeo_johnson_transform.png
│   └── visualisation_report.pdf
├── report.pdf                     # Final report
├── report.tex                     # LaTeX source
├── pledge.txt                     # Signed honor pledge
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── pyproject.toml                 # Project configuration
```

---

## 🚀 Setup Instructions

### Prerequisites
For now:
- Python >= 3.12
- uv (recommended) or pip

### Installation

```bash
# Clone or extract the project
cd Bayramov_Bayram_hw3_not_pure

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

### Running the Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/hw3_analysis.ipynb
# or
jupyter lab notebooks/hw3_analysis.ipynb
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=term

# Run specific test
uv run pytest tests/test_preprocessing.py::test_knn_imputer_basic -v
```

### Linting & Type Checking

```bash
# Run linter
uv run ruff check .

# Run type checker
uv run ty check .
```

---

## 🔧 Custom Implementations

### Imputers (`src/imputers.py`)

| Class | Features |
|-------|----------|
| **SimpleImputer** | Mean/median imputation, vectorized broadcasting, NaN fallback |
| **KNNImputer** | NaN-aware Euclidean distance, KD-Tree optimization, global mean fallback |

### Encoders (`src/encoders.py`)

| Class | Features |
|-------|----------|
| **OneHotEncoder** | Drop='first', vectorized broadcasting, optional sparse output |
| **TargetEncoder** | CV-safe with internal KFold, Laplace smoothing, numpy optimization |

### Scalers (`src/scalers.py`)

| Class | Features |
|-------|----------|
| **StandardScaler** | Mean/std standardization, inverse_transform |
| **MinMaxScaler** | Feature range [0,1], optional clipping |
| **RobustScaler** | Median/IQR scaling, inverse_transform |

### Feature Creation (`src/feature_creation.py`)

| Class | Features |
|-------|----------|
| **PolynomialFeatures** | Degree 2 with bias, interaction_only option, caching |
| **KBinsDiscretizer** | Uniform/quantile binning, one-hot encoding |

---

## 📈 Key Findings

1. **Class Disparity:** 1st class passengers were more likely to survive than 3rd class (63.0% vs 24.2%)

2. **Children First:** Children under 16 had 59.0% survival vs 38.2% for adults (χ²=15.56, p=0.00008)

3. **Wealth Matters:** Survivors paid $48.40 average fare vs $22.12 for non-survivors (t=-7.94, p<0.0001)

4. **Gender + Class:** 1st class women had 96.8% survival vs 13.5% for 3rd class men (gap of 83.3 percentage points)

5. **Interaction Effect:** The benefit of higher fare depends on class (LR=7.38, p=0.025)

---

## 🎯 Learning Objectives Achieved

| Objective | Demonstrated |
|-----------|--------------|
| **LO1** | Missing data handling (MCAR/MAR/MNAR), imputation, leakage prevention |
| **LO2** | Outlier treatment (winsorization), CV-safe categorical encoding |
| **LO3** | Feature engineering (scaling, polynomials, binning, Yeo-Johnson) |
| **LO4** | Data visualization (Anscombe, Titanic EDA, Tufte's principles) |


---

## 📚 Datasets

- [`Titanic dataset`](https://www.kaggle.com/c/titanic)
- [`California Housing`](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [`Anscombe's Quartet`](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)