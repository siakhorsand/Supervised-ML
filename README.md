# Comparative Analysis of Supervised Machine Learning Models

## Background

This research evaluates the performance of several supervised machine learning models across multiple classification datasets. We implement and compare Random Forest, Support Vector Machine (SVM), and Logistic Regression algorithms to identify which models perform best under different data characteristics. This analysis provides insights into model selection criteria for various classification tasks.

## Introduction

Choosing the appropriate machine learning model for a given dataset remains a challenging problem in data science. Different algorithms have varying strengths and weaknesses depending on data characteristics such as dimensionality, class balance, and feature relationships. This project examines three popular supervised learning algorithms across three distinct datasets to identify patterns in model performance.

## Datasets

Our analysis employs three widely-used classification datasets:

1. **Credit Approval Dataset**: Binary classification task to predict credit approval decisions
2. **Tic-Tac-Toe Dataset**: Classification of game outcomes based on board configurations
3. **Wisconsin Breast Cancer Dataset**: Binary classification of tumor diagnoses

## Model Performance Analysis

The detailed model performance analysis is available in the Jupyter notebook (`notebook.ipynb`). Key findings include:

- Logistic Regression performs best on the Credit Approval dataset (72.6% accuracy)
- SVM excels on the Tic-Tac-Toe dataset (98.3% accuracy) 
- Random Forest achieves highest accuracy on the Breast Cancer dataset (96.4%)

All visualizations from the analysis have been externalized to the `imgs/` directory to improve notebook performance and facilitate version control.

## Implementation

- `notebook.ipynb`: Main analysis notebook with model comparisons
- `data/`: Contains the three datasets used in the analysis
- `imgs/`: Externalized visualizations from the notebook
- `docs/`: Additional documentation and resources

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
seaborn
```

## Visualization Analysis

Our analysis produced several key visualizations to help understand model performance:

### Accuracy Comparison Across Datasets

The accuracy comparison shows that all models performed best on the Wisconsin Breast Cancer and Tic-Tac-Toe datasets, with more modest performance on the Credit Approval dataset. This suggests that the Credit dataset may contain more noise or complex relationships that are harder to model.

### ROC Curve Analysis

The ROC curves reveal that for the Credit Approval dataset, Logistic Regression achieved the best balance between true positive and false positive rates. For the Wisconsin dataset, all models showed excellent discrimination ability with AUC values above 0.95.

### Confusion Matrices

The confusion matrices highlight that:
- For Credit Approval, models had higher precision for approvals than rejections
- For Tic-Tac-Toe, both SVM and Logistic Regression correctly classified nearly all instances
- For Breast Cancer, Random Forest had the fewest false negatives, which is particularly important in medical diagnostics

## Cross-Model Analysis

Analyzing performance across all datasets reveals distinct patterns:

1. **Logistic Regression** performs well when the relationship between features and target is approximately linear, as seen in the Credit Approval dataset.

2. **SVM** excels when clear decision boundaries exist between classes, as demonstrated by its performance on the Tic-Tac-Toe dataset.

3. **Random Forest** shows robust performance across diverse datasets, particularly excelling on the Breast Cancer dataset where it captured complex feature interactions.

## Conclusion

Our comparative analysis reveals that no single model consistently outperforms others across all datasets. The optimal model choice depends on dataset characteristics:

- For datasets with linear relationships, Logistic Regression offers a good balance of performance and interpretability
- For datasets with clear but non-linear decision boundaries, SVM provides excellent classification power
- For datasets with complex feature interactions, Random Forest offers robust performance without requiring extensive feature engineering

This analysis highlights the importance of testing multiple models on a given dataset rather than assuming a "one-size-fits-all" approach to classification tasks.

## Implementation Details

The analysis was implemented in Python using scikit-learn, with visualizations created using matplotlib and seaborn. The code for model training, evaluation, and visualization is available in the EnhancedFinalProj.ipynb notebook in the COGS-118A directory.

For better performance and version control, visualizations were extracted as external files using our custom extraction tool (`extract_visualizations.py`). This approach allows for faster notebook loading and easier tracking of visualization changes. 