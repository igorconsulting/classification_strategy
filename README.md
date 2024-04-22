# Classification Strategy Ensemble Model

## Introduction
This repository contains a Python-based machine learning project that demonstrates the use of ensemble learning techniques for classification tasks. It utilizes a `StackingModel` class to combine predictions from various base models and uses a meta-model to improve prediction accuracy.

## Libraries Used
- Pandas, NumPy: For data manipulation and numerical operations.
- Scikit-learn: For machine learning model functions.
- XGBoost, LightGBM, CatBoost: Gradient boosting frameworks for building ensemble models.
- Seaborn, Matplotlib, Plotly: For data visualization.
- Warnings: To handle warnings in the code.

## Base Models
The ensemble uses the following base models:
- Random Forest
- Gradient Boosting
- AdaBoost
- Support Vector Machine (SVC)
- CatBoostClassifier
- LightGBMClassifier

These models are chosen for their diverse learning strategies, and their predictions are combined to create a robust classification system.

## Stacking Model
The `StackingModel` class is a custom-built ensemble strategy that leverages the strengths of multiple base models and a meta-model for prediction.

### Key Methods
- `add_base_models(models)`: Add base models to the ensemble.
- `set_meta_model(model)`: Set the meta-estimator that will be trained on the out-of-fold predictions of the base models.
- `get_oof(model, params, x_train, y_train, x_test, n_splits)`: Generate out-of-fold predictions for a given model.
- `get_oof_list(x_train, y_train, x_test)`: Compute and store out-of-fold predictions for all base models.
- `meta_fit(meta_x_train, y_train)`: Fit the meta-model using the prepared meta training dataset.
- `predict(X)`: Predict using the meta-model on the provided dataset.
- `get_feature_importance()`: Retrieve feature importance from base models.
- `metric_evaluation(metric, y_pred, y_test)`: Evaluate the model using the specified metric.

## Evaluation Metrics
The following metrics are used to evaluate model performance:
- Accuracy Score
- Precision Score
- Recall Score
- F1 Score

These metrics provide a comprehensive view of the model's classification capabilities.

## Visualization
Data visualization plays a key role in understanding the performance and characteristics of the models. We use various plotting libraries to visualize:
- Feature importance
- Correlation of feature importances between base models

## Usage
To use this ensemble strategy in your projects:
1. Initialize the `StackingModel`.
2. Add your selection of base models.
3. Set a meta-model.
4. Prepare your dataset and perform train/test splits.
5. Use the `get_oof` method to get out-of-fold predictions for base models.
6. Fit the meta-model with `meta_fit`.
7. Predict on new data using the `predict` method.

Please refer to the provided example notebook for a detailed walkthrough of the implementation.

## Contributions
Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License
This project is open-sourced under the MIT license.
