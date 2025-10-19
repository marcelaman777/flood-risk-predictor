# Flood Prediction - Comprehensive EDA, Multi-Model Regression & Bayesian-Optimized CatBoost

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-brightgreen)
![Model](https://img.shields.io/badge/Model-Multi%20Algorithm%20Ensemble-red)
![Analysis](https://img.shields.io/badge/Analysis-EDA%20%2B%20Feature%20Engineering-yellowgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

This comprehensive project implements a complete flood prediction pipeline featuring extensive exploratory data analysis, systematic benchmarking of multiple regression algorithms, and advanced Bayesian-optimized CatBoost modeling. The goal is to predict regional flood probability based on 20 environmental and infrastructural factors, achieving state-of-the-art performance through rigorous experimentation and sophisticated feature engineering.

**Competition Context**: This project is part of Kaggle's Playground Series S4E5 competition, where datasets were generated from a deep learning model trained on the original Flood Prediction Factors dataset.

## 🎯 Objective

Predict flood probability (continuous target ranging from 0.285 to 0.725) through systematic comparison of linear and tree-based regression models, culminating in an advanced Bayesian-optimized CatBoost implementation with innovative feature engineering.

## 📊 Dataset Features

### 20 Flood Prediction Factors (Categorical Features 0-19)
- **MonsoonIntensity** - Intensity of monsoon patterns
- **TopographyDrainage** - Drainage characteristics of topography
- **RiverManagement** - River management quality
- **Deforestation** - Level of deforestation
- **Urbanization** - Degree of urbanization
- **ClimateChange** - Climate change impact
- **DamsQuality** - Quality of dam infrastructure
- **Siltation** - Siltation levels
- **AgriculturalPractices** - Agricultural methods
- **Encroachments** - Land encroachment extent
- **IneffectiveDisasterPreparedness** - Disaster readiness effectiveness
- **DrainageSystems** - Drainage system quality
- **CoastalVulnerability** - Coastal area vulnerability
- **Landslides** - Landslide risk
- **Watersheds** - Watershed conditions
- **DeterioratingInfrastructure** - Infrastructure deterioration
- **PopulationScore** - Population density score
- **WetlandLoss** - Wetland loss extent
- **InadequatePlanning** - Planning adequacy
- **PoliticalFactors** - Political influences

### Target Variable
- **FloodProbability**: Continuous probability value ranging from 0.285 to 0.725

## 📊 Evaluation Framework

**R² Score (Coefficient of Determination)**
- **Interpretation Scale**: 
  - 1.0: Perfect predictive accuracy
  - 0.0: Equivalent to mean prediction
  - <0.0: Inferior to simple averaging

## 🏗️ Project Architecture

### Phase 1: Comprehensive EDA & Multi-Model Benchmarking
**File**: `flood-prediction-eda-linear-tree-based-models.ipynb`  
*Complete statistical analysis, distribution profiling, and systematic comparison of 15+ regression models across linear and tree-based families*

### Phase 2: Advanced Feature Engineering & Bayesian-Optimized CatBoost
**File**: `flood-prediction-catb-bayes-opt-target-enc.ipynb`  
*Sophisticated feature engineering with 24 statistical features, Gaussian process target encoding, and Bayesian-optimized CatBoost regression*

## 🔍 Comprehensive EDA Methodology

### 1. Data Profiling & Quality Assessment
- **Dataset Scale**: 1,117,957 training samples, 745,305 test samples
- **Feature Analysis**: All 20 features categorical (0-19 integer range)
- **Data Integrity**: Zero missing values, no duplicate records
- **Memory Optimization**: Strategic dtype selection (uint8 for features, float64 for target)

### 2. Advanced Distribution Analysis
**Statistical Insights**:
- **Poisson Distribution**: All features follow λ=5 distribution, indicating synthetic generation
- **Distribution Consistency**: Train/test sets statistically identical (Anderson-Darling validation)
- **Central Tendency**: Median = 5 across all features, mode typically matches median
- **Quartile Analysis**: Identical upper quartiles across features, unusual but consistent

### 3. Target Variable Deep Dive
**FloodProbability Characteristics**:
- **Value Range**: 0.285 to 0.725 (continuous probability)
- **Value Cardinality**: Only 83 unique values from 1.1M+ records
- **Distribution Shape**: Approximately normal (bell curve distribution)
- **Statistical Moments**: Mean = 0.504, Median = 0.505, Mode = 0.490

### 4. Correlation & Relationship Mapping
**Multi-Method Correlation Analysis**: 
- Assuming that all features are actually ordinal:
  - **Feature-Target Correlations (Spearman's method)**: Minimal individual correlations (all < 0.19)
  - **Feature Inter-correlations (Spearman's method)**: Negligible relationships between predictors
- **Key Discovery**: Sum of features shows exceptional correlation (R² = 0.844) with target

### 5. Bivariate & Multivariate Analysis
**Pattern Recognition**:
- **Monotonic Relationships**: Consistent increase in flood probability with rising feature values
- **Ordinal Nature**: Clear ordinal relationships despite categorical representation

## 🛠️ Advanced Feature Engineering

### 1. Statistical Feature Synthesis (24 New Features)

**Distribution Statistics**:
- `fmax`, `fmin`: Range indicators across feature set
- `fsum`: Aggregate risk score (highly predictive)
- `fmedian`: Central tendency measure
- `fstd`: Variability indicator
- `fmean+std`, `fmean-std`: Mean-variance boundaries

**Quantile Analysis**:
- Comprehensive percentile features (`fq10th` to `fq90th`)
- Interquartile range components (`fq25th`, `fq75th`)
- Distribution shape descriptors

**Higher-Order Moments**:
- `fsq_sum`, `fcb_sum`: Polynomial aggregates
- `fskew`, `fkurtosis`: Distribution shape metrics
- `fgeom_mean`, `fharm_mean`: Alternative central tendency measures

### 2. Innovative Target Encoding

#### Gaussian Process Encoding

- Implemented customTargetEncodingTransformer class that leverages Gaussian Process Regression (GPR) to transform the aggregated sum of original features into a continuous, probabilistically-encoded representation. This advanced encoding method captures the complex, non-linear relationship between feature aggregates and flood probability with built-in uncertainty quantification.

```python
class TargetEncodingTransformer(TransformerMixin, BaseEstimator):
    '''Advanced target encoding using Gaussian Process Regression for smooth encoding'''
    
    def fit(self, X, y=None):
        # Calculate mean flood probability per fsum value
        # Implement GPR for robust encoding of unseen values
        # Create smooth encoding surface
```

#### Encoding Feature Suite

- **ftarget_enc**: GPR-based probabilistic encoding
- **fesp_q25th**: Expected value using 25th percentile weighting  
- **fesp_q75th**: Expected value using 75th percentile weighting

**Final Feature Space**: 47 dimensions (20 original + 24 statistical + 3 encoding)

## 📈 Comprehensive Model Benchmarking

### 1. Linear Models Performance Comparison (cross-validated)

| Model | CV R² | Categorical Features | Numerical Features |
|-------|-------|----------------------|--------------------|
| **Ridge** | 0.8667 | Original features + \[ fsum, fmedian, fmin, fmax \] | \[ fstd, fmean+std, fmean-std \] |
| **Ridge** | 0.8665 | Original features + \[ fsum, fmedian, fmin, fmax \] | |
| **Ridge** | 0.8665 | Original features + \[ fsum \] | |
| **Ridge** | 0.8460 | Original features | stats features |
| **Ridge** | 0.8456 | Original features | |
| **Linear** | 0.8456 | Original features | |
| **Ridge** | 0.8451 | | Original features |
| **Linear** | 0.8451 | | Original features |
| **Lasso** | -3.1e-06 | Original features | |
| **Lasso** | -3.1e-06 | | Original features |

### 2. Tree-Based Models Performance Hierarchy (cross-validated)

| Model | CV R² | Numerical Features |
|-------|-------|----------------|
| **Bayesian CatBoost** | 0.8693 | Advanced Feature Engineering |
| **LightGBM Tuned** | 0.8690 | Original + stats Features |
| **CatBoost** | 0.8689 | Original + stats features |
| **CatBoost Tuned** | 0.8688 | Original + stats features |
| **HistGradientBoosting Tuned** | 0.8687 | Original + stats features |
| **LightGBM** | 0.8687 | Original + stats Features |
| **HistGradientBoosting** | 0.8686 | Original + stats features |
| **XGBoost Tuned** | 0.8686 | Original + stats features |
| **XGBoost** | 0.8686 | Original + stats features |
| **CatBoost** | 0.8675 | Original features + \[ fsum \] |
| **LightGBM** | 0.8672 | Original Features + \[ fsum \] |
| **XGBoost** | 0.8672 | Original features + \[ fsum \] |
| **HistGradientBoosting** | 0.8671 | Original features + \[ fsum \] |
| **CatBoost** | 0.8463 | Original features |
| **XGBoost** | 0.8102 | Original features |
| **LightGBM** | 0.7665 | Original Features |
| **HistGradientBoosting** | 0.7664 | Original features |

## 🚀 Bayesian-Optimized CatBoost Implementation

### 1. Model Architecture

**CatBoost Regressor Configuration:**:

```python
FIXED_PARAMS = {
    'boosting_type': 'Plain',
    'grow_policy': 'Lossguide', 
    'task_type': 'GPU'
}

TUNING_PARAMS = {
    'regressor__n_estimators': (1000, 3000),
    'regressor__max_depth': (6, 10),
    'regressor__num_leaves': (31, 511),
    'regressor__learning_rate': (0.01, 0.03),
    'regressor__min_child_samples': (1, 120),
    'regressor__reg_lambda': (0.00001, 100.0, 'log-uniform')
}
```

### 2. Bayesian Optimization Strategy

#### BayesSearchCV Configuration:
- Iterations: 50 Bayesian optimization rounds
- Cross-Validation: 5-fold with random state 42
- Scoring: R² metric
- Random State: 0 for reproducible optimization

#### Training Strategy:
- 5-Fold Cross-Validation with consistent random state
- GPU Acceleration for efficient training
- Ensemble Prediction by averaging fold predictions

### 3. Performance Results

#### Bayesian Optimization Results

**Best Hyperparameters Found**:

```python
{
    'learning_rate': 0.01,
    'max_depth': 10, 
    'min_child_samples': 93,
    'n_estimators': 1000,
    'num_leaves': 372,
    'reg_lambda': 1e-05
}
```

**Cross-Validation Performance**:

| Fold | Validation R² |
|------|---------------|
| 1 | 0.8691 |
| 2 | 0.8697 |
| 3 | 0.8691 |
| 4 | 0.8693 |
| 5 | 0.8693 |

**Final Model Performance**:
- **Mean Validation R²**: 0.8693
- **Best Bayesian Search R²**: 0.8693
- **Training Time**: ~4 hours (GPU accelerated)
- **Performance Stability**: Exceptional cross-fold consistency

### 4. Feature Importance Insights

**Top Predictive Features**:
- **ftarget_enc** - Gaussian process target encoding (dominant)
- **fsum** - Sum of all features (highly correlated with target)

**Key Insight**: Engineered statistical features and target encoding provide significantly more predictive power than original features alone.

## 🏆 Model Performance Summary

1. **🥇 Bayesian CatBoost**: 0.8693 R² (Champion Model)
2. **🥈 LightGBM Tuned**: 0.8690 R² (Strong Contender)
3. **🥉 CatBoost**: 0.8689 R² (Feature Engineering Showcase)

## 🛠️ Technical Implementation

### Libraries Used
- **pandas, numpy** - Data manipulation and numerical computing
- **matplotlib, seaborn** - Visualization and plotting
- **scipy, statsmodels** - Statistical testing and analysis
- **scikit-learn** - Preprocessing, metrics, and pipeline construction
- **xgboost, lightgbm, catboost, HistGradientBoostingRegressor** - Gradient boosting implementations
- **skopt** - Bayesian optimization library
- **scikit-learn GPR** - Gaussian Process Regression for target encoding

### Key Features:
- Gaussian Process transformation for categorical feature engineering
- Stratified cross-validation for robust evaluation
- GPU-accelerated training with CatBoost

## 🔮 Future Work

### Model Improvements
- **Advanced Ensembles**: Stacking and blending techniques
- **Neural Networks**: DNNs for capturing complex interactions
- **Feature Interactions**: Creating interaction terms

### Technical Enhancements
- SHAP Analysis: For model interpretability
- Cross-Validation: Group-based cross-validation
- Model Calibration: Ensuring probability calibration
- Deployment Pipeline: API for real-time predictions

## 📚 References

### Competition & Data
- [Kaggle Competition: Playground Series S4E5](https://www.kaggle.com/competitions/playground-series-s4e5)
- [Original Dataset: Flood Prediction Factors](https://www.kaggle.com/datasets/brijlaldhankour/flood-prediction-factors)

## Technical References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/) 
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [Anderson-Darling K-Sample Test](https://faculty.washington.edu/fscholz/Papers/ADk.pdf)
- [Bayesian Optimization Principles](https://arxiv.org/abs/1807.02811)
- [Gaussian Process Regression](https://gaussianprocess.org/gpml/)

## 🌟 Achievement Summary

This project demonstrates a complete, production-grade machine learning pipeline achieving competitive performance (R² = 0.8693) through systematic methodology, rigorous experimentation, and advanced optimization techniques. The comprehensive approach from exploratory analysis to optimized modeling provides both practical solutions and methodological insights for regression challenges in environmental prediction domains.

## 🚀 Getting Started

### Prerequisites

```python
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels xgboost lightgbm catboost
```

### Usage

#### 1. Clone this repository:

```python
git clone https://github.com/marcelaman777/flood-risk-predictor.git
```

#### 2. Navigate to the project directory:

```python
cd flood-risk-predictor
```

#### 3. Run the main notebook:

```python
# Complete pipeline: EDA + Feature Engineering + Model Training
jupyter notebook flood-prediction-eda-linear-tree-based-models.ipynb
```

#### 4. Alternative: Run individual components (if separated):

```python
# Exploratory Data Analysis & Statistical Testing
flood-prediction-eda-linear-tree-based-models.ipynb

# Model Training & Ensemble Optimization
flood-prediction-eda-linear-tree-based-models.ipynb

# Feature Engineering & Gaussian Process Encoding
flood-prediction-catb-bayes-opt-target-enc.ipynb 
```

## 📁 Project Structure

```python
flood-risk-predictor/
├── .gitignore                                            # Git ignore rules
├── LICENSE                                               # MIT License file
├── README.md                                             # Project documentation
├── flood-prediction-eda-linear-tree-based-models.ipynb   # Main notebook: Comprehensive EDA & Multi-Model Benchmarking
└── flood-prediction-catb-bayes-opt-target-enc.ipynb      # Advanced Feature Engineering & Bayesian-Optimized CatBoost
```

