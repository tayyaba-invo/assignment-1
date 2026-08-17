# Observations and Conclusions

## Dataset and class imbalance
The Telco Customer Churn dataset contains more than 1,000 records and includes both numerical and categorical features. The target is binary churn.

The observed target distribution is approximately:
- No Churn: 73.46%
- Churn: 26.54%

This is a moderately imbalanced target. Therefore, accuracy alone is not sufficient. Precision, recall, F1-score, confusion matrices and ROC-AUC should also be considered.

## Preprocessing decisions
`customerID` was removed because it is an identifier and does not provide useful predictive information. `TotalCharges` was converted from text to numeric, with invalid values converted to missing values. Numerical variables are median-imputed and scaled, while categorical variables are imputed and one-hot encoded.

The preprocessing is inside sklearn pipelines so transformations are learned from the training data rather than from the validation/test sets.

## KNN experiment
Two K values were tested: K=3 and K=7. The final comparison table in the notebook should be used to determine which K gives better validation F1-score/ROC-AUC. A larger K generally produces a smoother decision boundary, while a smaller K is more sensitive to individual training examples.

## Decision Tree depth
A shallow tree (`max_depth=3`) and a deeper tree (`max_depth=15`) were compared.

The existing experiment showed that the deep tree achieved much higher training accuracy than validation accuracy, which is a clear sign of high variance/overfitting. The shallow tree had a much smaller training-validation gap.

A `max_depth=1` tree is also included as the underfitting/high-bias comparison. Its training and validation performance can be compared in the generated table.

## Decision Tree vs Random Forest
A single Decision Tree is easy to interpret because its decisions can be followed through tree branches. Random Forest combines many trees, usually improving robustness and reducing the variance of a single tree, at the cost of interpretability.

## Random Forest vs Gradient Boosting
Random Forest and Gradient Boosting are both ensemble methods, but they build ensembles differently. The validation and final test tables should be used to compare their F1-score, ROC-AUC and timing.

## SVM scaling experiment
The assignment requires an SVM comparison with and without scaling. The completed notebook trains both versions. Since SVMs are sensitive to feature magnitude, the scaled version is expected to provide a fairer representation when numerical variables have different ranges. The actual validation metrics should be taken from the notebook output.

## Generalization
The validation set is used for model selection. The test set is used only for the final unbiased comparison. The best generalizing model should therefore be selected using the final test results only after the validation comparison has been completed.

## Interpretability
If explainability is the main requirement, a Decision Tree is the most straightforward choice among the tested algorithms because its decision rules can be inspected directly.

## Predictive performance
If predictive performance is the primary objective, choose the model with the strongest final-test F1-score/ROC-AUC, while considering the business importance of false positives versus false negatives.

## Speed
Inference time is measured for every model. The notebook automatically identifies the fastest inference model rather than relying on manually entered timings.

## Bias and variance
- High bias/underfitting: represented by the `max_depth=1` Decision Tree if its training and validation scores are both low.
- High variance/overfitting: represented by the `max_depth=15` Decision Tree because its training score is much higher than its validation score.

## Final model-selection rule
The completed notebook prints:
1. Best validation model by F1-score.
2. Best validation model by ROC-AUC.
3. Best final test model by F1-score.
4. Best final test model by ROC-AUC.
5. Fastest inference model.

Use these automatically generated values in the final written conclusion rather than manually copying outdated metrics.
