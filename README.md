# Advanced Data Analytics Assessment

A comprehensive Python-based data analytics pipeline for demand forecasting and generation cost prediction using machine learning ensemble models.

**Pipeline:** Data Preparation → Exploratory Data Analysis → ML Model Training & Comparison

## 📋 Project Overview

This project implements a complete ML workflow to analyze relationships between energy demand patterns and generation costs, ultimately training three ensemble models (Random Forest, Gradient Boosting, Extra Trees) with decision-level evaluation.

### Key Features
- **Data Preparation:** Merges multiple data sources (demand, generation costs, plant info)
- **EDA:** Correlation analysis, cost distribution, demand patterns visualization
- **ML Models:** Optimized hyperparameter tuning with GroupKFold cross-validation
- **Performance:** ~80 seconds total runtime (optimized from ~150+ seconds)
- **Benchmarking:** Decision-level RMSE scoring on demand-group selection

## 📦 Requirements

Python 3.8+

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/demilade2001/energy-cost-prediction.git
cd energy-cost-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare input data files (place in `~/Downloads/`):
   - `demand.csv` - Demand data with columns: `Demand ID`, `DF_region`
   - `generation_costs.csv` - Cost data with columns: `Demand ID`, `Plant ID`, `Cost_USD_per_MWh`
   - `plants.csv` - Plant info with columns: `Plant ID`, `Region`

### Running the Pipeline

**Option 1: Run all steps (recommended)**
```bash
python MASTER_RUNNER.py
```

**Option 2: Run without EDA visualizations**
```bash
python MASTER_RUNNER.py --no-eda
```

**Option 3: Run individual scripts**
```bash
python "Preparation Solution.py"
python EDA.py
python "ML MODEL.py"
```

## 📊 Output

### Generated Files
- `cleaned merged data.csv` - Preprocessed dataset
- `Model_Comparison_Table.csv` - ML model results and metrics

## 🔧 Project Structure

```
.
├── MASTER_RUNNER.py
├── Preparation Solution.py
├── EDA.py
├── ML MODEL.py
├── Model_Comparison_Table.csv
├── README.md
└── requirements.txt
```

## 🎯 Model Performance

| Model | CV Decision RMSE | Test Decision RMSE |
|-------|------------------|-------------------|
| Random Forest | 6.13 | 5.56 |
| Gradient Boosting | 5.32 | 5.54 |
| Extra Trees | 4.08 | 3.79 |
| Plant-Mean Baseline | N/A | 9.33 |

## 📝 Author

Demilade Adeniyi

## 📄 License

Add your license here.
