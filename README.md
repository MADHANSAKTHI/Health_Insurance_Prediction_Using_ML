# Health Insurance Premium Charges Prediction

Predicting individual health insurance charges from age, BMI, smoking status and other personal details using **Linear Regression**.

The project walks through the full workflow in a single notebook: data cleaning, exploratory data analysis (EDA), preprocessing, building three models, comparing them, and making predictions for new customers.

## Problem Statement

Insurers need to estimate how much a person is likely to cost them so premiums can be set fairly. The goal here is to predict **insurance charges** for an individual using their demographic and lifestyle information, and to understand which factors drive cost the most.

## Dataset

The dataset contains **1,338 records** and 7 columns. One exact duplicate row was found and removed, leaving **1,337 unique records**. There are no missing values.

| Column | Description |
|--------|-------------|
| `age` | Age of the person (18 to 64) |
| `sex` | Gender (male / female) |
| `bmi` | Body mass index |
| `children` | Number of children covered by the insurance |
| `smoker` | Smoker status (yes / no) |
| `region` | Residential region (northeast, northwest, southeast, southwest) |
| `charges` | Insurance cost billed (**target variable**) |

## Project Workflow

1. **Data loading**: read `insurance_prediction.csv` with pandas.
2. **Data overview**: check shape, data types, summary statistics and missing values.
3. **Duplicate handling**: find and drop duplicate rows.
4. **EDA**: univariate and bivariate analysis with histograms, count plots, box plots, a pair plot and a correlation heatmap.
5. **Preprocessing**: one-hot encode `sex`, `smoker` and `region` (with `drop_first=True`), then split the data 70% train / 30% test.
6. **Modeling**: train three Linear Regression models with increasing feature sets.
7. **Evaluation**: compare models using R² and MSE on both training and test data.
8. **Prediction**: use the best model to predict charges for new individuals.

## Key Insights from EDA

- **Smokers pay far more** than non-smokers, regardless of gender.
- **Age** has a positive relationship with charges (correlation about 0.30).
- **BMI** has a weaker positive relationship with charges (about 0.20), which is stronger for smokers.
- Charges rise with 1 to 2 children, then level off.
- Charges are broadly similar across regions, with the southeast showing more variation and some very high values.
- Charges are right-skewed, with a number of high-cost outliers.

## Models and Results

| Model | Features | Train R² | Test R² | Train MSE | Test MSE |
|-------|----------|----------|---------|-----------|----------|
| Model 1 | Age only | 0.082 | 0.097 | 1.25e+08 | 1.55e+08 |
| Model 2 | Age + BMI | 0.099 | 0.140 | 1.22e+08 | 1.47e+08 |
| **Model 3** | **All features** | **0.736** | **0.772** | **3.58e+07** | **3.89e+07** |

Model 3 explains about **77% of the variation** in charges on unseen data, and its train and test scores are close, which suggests it is not overfitting.

**Reading the equation:** being a smoker adds roughly **$22,900** to the predicted charge, far more than any other factor. Each extra year of age adds about $251, and each extra BMI point about $328.

## Business Recommendations

- **Targeted pricing:** adjust premiums based on age and BMI.
- **Smoking incentives:** reward non-smokers and encourage quitting.
- **Wellness programs:** promote healthy living to reduce long-term health risks.
- **Regional strategy:** use regional pricing cautiously, only where cost differences justify it.
- **Marketing focus:** offer plans suited to both low-risk and high-risk clients.

## Tech Stack

- Python 
- Jupyter Notebook / Google Colab
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
