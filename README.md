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
  - **Feature-Target Correlations (Spearman's method)**: Minimal individual correlations (all < 0.18)
  - **Feature Inter-correlations (Spearman's method)**: Negligible relationships between predictors
- **Key Discovery**: Sum of features shows exceptional correlation (R² = 0.844) with target

### 5. Bivariate & Multivariate Analysis
**Pattern Recognition**:
- **Monotonic Relationships**: Consistent increase in flood probability with rising feature values
- **Ordinal Nature**: Clear ordinal relationships despite categorical representation

## 🛠️ Advanced Feature Engineering

### Statistical Feature Synthesis (24 New Features)

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

### Innovative Target Encoding

#### Gaussian Process Encoding

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

### Linear Models Performance Comparison (cross-validated)

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

### Tree-Based Models Performance Hierarchy (cross-validated)

| Model | CV R² | Numerical Features |
|-------|-------|----------------|
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

### Feature Engineering Impact Analysis

| Feature Strategy | R² Improvement | Relative Gain |
|------------------|----------------|---------------|
| **Base Features Only** | 0.8632 | Reference |
| **+ Statistical Features** | +0.0055 | +0.64% |
| **+ Sum Feature Only** | +0.0033 | +0.38% |
| **+ Target Encoding** | +0.0005 | +0.06% |
| **Full Feature Suite** | **0.8693** | **+0.71%** |

### Hyperparameter Optimization Results

| Optimization Method | Best R² | Iterations | Time Investment |
|---------------------|----------|------------|-----------------|
| **Default Parameters** | 0.8688 | - | Minimal |
| **Random Search** | 0.8689 | 50 | Moderate |
| **Grid Search** | 0.8689 | 100+ | High |
| **Bayesian Optimization** | **0.8693** | 50 | High (GPU) |

## 🚀 Bayesian-Optimized CatBoost Implementation

### Optimization Architecture

**Search Space Definition**:

```python
TUNING_PARAMS = {
    'regressor__n_estimators': (1000, 3000),
    'regressor__max_depth': (6, 10),
    'regressor__num_leaves': (31, 511),
    'regressor__learning_rate': (0.01, 0.03),
    'regressor__min_child_samples': (1, 120),
    'regressor__reg_lambda': (0.00001, 100.0, 'log-uniform')
}
```

