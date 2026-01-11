# Mobile-Money-Fraud-Detection

A comprehensive machine learning solution for detecting fraudulent transactions in mobile money services, achieving 69.3% fraud detection rate with only 0.037% false positive rate.

## Project Overview

This project implements an end-to-end fraud detection system using the PaySim synthetic financial dataset, which simulates mobile money transactions based on real financial service data from Africa. The system processes 6.3M+ transactions to identify fraudulent patterns while minimizing false alarms.

### Key Results

- **Detection Rate**: 69.3% (1,138 / 1,643 frauds caught)
- **False Positive Rate**: 0.037% (471 / 1,270,881 legitimate transactions flagged incorrectly)
- **Precision**: 70.7% of flagged transactions are actually fraudulent
- **Review Workload**: Only 1,609 transactions flagged for manual review (0.13% of total)
- **ROI**: 20,765x return on investment
- **Net Benefit**: $1.67 billion in fraud prevented (test set)
- **Production Model**: LightGBM with optimized threshold (0.970)

## Dataset

**Source**: [PaySim Mobile Money Transaction Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

### Dataset Overview
- **Size**: 6,362,620 transactions over 30 days (744 hourly steps)
- **Fraud Rate**: 0.13% (8,213 fraudulent transactions)
- **Class Imbalance**: 1:773 (fraud to legitimate ratio)
- **Origin**: Synthetic data generated using PaySim simulator based on real mobile money transactions from an African mobile financial service
- **Scale**: 1/4 scale of original private dataset

### Key Characteristics
- **Transaction Types**: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER
- **Critical Finding**: 100% of fraud occurs only in TRANSFER and CASH-OUT transaction types
- **Fraud Pattern**: Fraudulent agents take control of customer accounts and attempt to empty funds by transferring to another account then cashing out

### Dataset Columns
- `step`: Unit of time (1 step = 1 hour, total 744 steps = 30 days)
- `type`: Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)
- `amount`: Transaction amount in local currency
- `nameOrig`: Customer who initiated the transaction
- `oldbalanceOrg`: Initial balance before transaction
- `newbalanceOrig`: Balance after transaction
- `nameDest`: Transaction recipient (M prefix = Merchant)
- `oldbalanceDest`: Recipient's initial balance
- `newbalanceDest`: Recipient's balance after transaction
- `isFraud`: Fraudulent transaction indicator (target variable)
- `isFlaggedFraud`: Business rule flag for transfers >200,000

### Important Note on Data Leakage ⚠️
**The dataset documentation explicitly warns:**
> "Transactions which are detected as fraud are cancelled, so for fraud detection these columns (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest) must not be used."

This project **correctly excludes these balance columns** to avoid data leakage and simulate real-world fraud detection where you only have transaction-time information available.

## Features

### Engineered Features (25 total)

The model uses only transaction-time data to avoid data leakage:

**Amount-based Features**
- Log transformation and squared amount
- Amount percentiles
- Deviation from customer averages

**Transaction Type Features**
- High-risk type indicators (TRANSFER, CASH_OUT)
- Transaction type encoding

**Behavioral Features**
- Customer transaction frequency
- Historical spending patterns
- Deviation from typical behavior

**Velocity Features**
- Transactions per time step
- Amount vs. period average

**Temporal Features**
- Hour of day, day of month
- Night transactions, weekend indicators

**Network Features**
- Destination frequency patterns
- Merchant vs. customer destinations

## Models Evaluated

| Model | ROC-AUC | Avg Precision | F2 Score | Frauds Caught | False Positives |
|-------|---------|---------------|----------|---------------|-----------------|
| Logistic Regression | 0.9655 | 0.1420 | 0.0539 | 1471/1643 | 128,517 |
| Random Forest | 0.9715 | 0.6876 | 0.6449 | 1079/1643 | 714 |
| Histogram GBM | 0.9820 | 0.6536 | 0.1507 | 1474/1643 | 40,866 |
| Ensemble (RF+HGB) | 0.9825 | 0.7008 | 0.6521 | 1060/1643 | 496 |
| XGBoost | 0.9845 | 0.7296 | 0.6820 | 1169/1643 | 829 |
| **LightGBM** | **0.9843** | **0.7389** | **0.6955** | **1138/1643** | **471** |

## Repository Contents

```
fraud-detection/
│
├── fraud_detection_analysis.ipynb    # Main Jupyter notebook with complete analysis
├── Synthetic_Financial_datasets_log.csv    # Dataset (6.3M transactions)
└── README.md    # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

# Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn
```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
imbalanced-learn>=0.10.0
```

## Usage

1. **Clone the repository**
2. **Install dependencies** (see above)
3. **Open the notebook**: `jupyter notebook fraud_detection_analysis.ipynb`
4. **Run all cells** to reproduce the complete analysis

The notebook includes:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training (6 different algorithms)
- Performance Evaluation
- Business Impact Analysis

## Methodology

### 1. Data Preparation
- **Critical**: Balance columns excluded to prevent data leakage
- Train/Validation/Test split: 70% / 10% / 20%
- Stratified sampling maintains fraud distribution

### 2. Feature Engineering
- 23 new features created from transaction metadata
- Focus on behavioral patterns and velocity metrics
- No use of post-transaction data (avoiding leakage)

### 3. Model Training
- SMOTE applied for class balancing (30% sampling strategy)
- Class weights used in tree-based models
- Validation set used for threshold optimization

### 4. Threshold Optimization
- F2 score maximization (prioritizes recall over precision)
- Optimal threshold: 0.970 for production deployment
- Validated on held-out validation set

### 5. Evaluation
- Final metrics on completely held-out test set
- Business impact analysis with cost-benefit calculations

## Business Impact

### Financial Metrics

- **Average Fraud Amount**: $1,467,967
- **Fraud Prevented**: $1.67 billion
- **Fraud Losses (Missed)**: $741 million
- **Manual Review Costs**: $80,450
- **Net Benefit**: $1.67 billion
- **Cost-Benefit Ratio**: $2.3 prevented for every $1 lost

### Operational Metrics

- **Detection Rate**: 69.3%
- **Precision**: 70.7% (of flagged transactions)
- **Review Workload**: ~1,609 transactions flagged daily
- **Cost per Detection**: $70.69

## Key Insights

1. **Transaction Type Risk**: TRANSFER and CASH_OUT account for 100% of fraud
2. **Amount Patterns**: Fraudulent transactions are 8.2x larger on average
3. **Heavy-Tailed Distribution**: Top 20% of frauds represent 74% of total fraud value
4. **Feature Importance**: Transaction count, risk type, and behavioral deviations are top predictors

## Production Deployment

### Recommended Implementation

1. Deploy LightGBM model with threshold = 0.970
2. Flag ~1,600 transactions daily for manual review
3. Expected to catch 69.3% of fraudulent transactions
4. Maintain false positive rate below 0.04%

### Monitoring Strategy

- Track false positive rate weekly (target: <0.5%)
- Monitor fraud detection rate (target: >75%)
- Review threshold quarterly based on fraud pattern changes
- Implement A/B testing with champion-challenger approach

### Future Improvements

- Add real-time velocity features
- Implement graph-based network analysis for fraud rings
- Deploy SHAP for model explainability and auditing
- Develop customer-specific risk profiles
- Experiment with deep learning models (LSTM for sequential patterns)

## Notebook Structure

The analysis is organized into clear sections:

1. **Data Loading & Quality Assessment** - Initial exploration and validation
2. **Exploratory Data Analysis (EDA)** - Fraud patterns by type, amount, and time
3. **Feature Engineering** - 23 new features from transaction metadata
4. **Data Preparation** - Train/validation/test split with stratification
5. **Baseline Models** - Logistic Regression with SMOTE
6. **Advanced Models** - Random Forest, Histogram GBM, Ensemble
7. **Gradient Boosting** - XGBoost and LightGBM with threshold optimization
8. **Model Comparison** - Comprehensive performance analysis
9. **Feature Importance** - Analysis across all models
10. **Business Impact Analysis** - ROI and cost-benefit calculations

## Visualizations Included

The notebook generates professional visualizations:
- Fraud distribution by transaction type
- Amount distribution analysis (legitimate vs fraudulent)
- ROC and Precision-Recall curves for all models
- Confusion matrices
- Feature importance heatmaps
- Threshold optimization plots
- Business impact dashboards

## Dataset Information

**Download**: The dataset can be obtained from [Kaggle - PaySim1 Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

**Note**: The CSV file is approximately 493 MB. You have two options:
1. **Download directly from Kaggle** and place it in the project root directory
2. **Use Git LFS** if you want to include it in version control:
   ```bash
   git lfs install
   git lfs track "*.csv"
   git add .gitattributes
   ```

Alternatively, you can add the CSV to `.gitignore` and provide download instructions in your repository.

## Acknowledgments

- **Dataset**: PaySim Mobile Money Transaction Simulator by Lopez-Rojas et al.
- **Inspiration**: Real-world fraud detection challenges in mobile money services
- **Tools**: scikit-learn, XGBoost, LightGBM, pandas, matplotlib, seaborn

## License

This project is open source and available for educational purposes.
## Connect - Taroon Ganesh (Data Science Grad - USC)

- **LinkedIn:** [linkedin.com/in/taroon-ganesh-27b83b171/](https://www.linkedin.com/in/taroon-ganesh-27b83b171/)
- **Email:** taroon2k@gmail.com

---
