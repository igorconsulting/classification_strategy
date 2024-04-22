# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import clear_output
from sklearn import model_selection,linear_model,metrics

# Model Evaluation Metrics
from sklearn.metrics import classification_report, accuracy_score, precision_score,recall_score,f1_score
from sklearn import preprocessing

#Model Selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV ,StratifiedKFold
from sklearn.linear_model  import LogisticRegression, RidgeClassifier


from sklearn.model_selection import KFold
import numpy as np


class StackingModel:
    """
    A class for stacking models which facilitates ensemble learning by using predictions of 
    several base models as input for a meta-model to improve prediction accuracy.

    Attributes:
        base_models (list): A list of tuples, each containing a model and its parameters.
        meta_model (object): The meta-estimator model that will be trained on the out-of-fold predictions of the base models.
        oof_train (numpy.ndarray): Array storing out-of-fold predictions for the training data.
        oof_test (numpy.ndarray): Array storing averaged predictions for the test data from all base models.
    """
    def __init__(self):
        self.base_models = []
        self.meta_model = None
        self.oof_train = None
        self.oof_test = None
        self.metric = accuracy_score

    def add_base_models(self, models):
        """ 
        Add multiple base models to the ensemble.
        
        Parameters:
            models (list): A list of tuples where each tuple contains a model class and its corresponding parameters.
        """
        self.base_models.extend(models)
        
    def get_base_models(self):
        """ Return the list of base models added to the ensemble. """
        return self.base_models

    def set_meta_model(self, model):
        """ Set the meta-model that will use the base models' out-of-fold predictions as features.
        
        Parameters:
            model (object): An instance of the model to be used as the meta-model.
        """
        self.meta_model = model
        
    def get_oof(self, model, params, x_train, y_train, x_test, n_splits=5):
        """
        Generate out-of-fold predictions for the given model and parameters using K-Fold cross-validation.
        
        Parameters:
            model (class): The machine learning model class to be used.
            params (dict): Dictionary of parameters to be used for model instantiation.
            x_train (DataFrame): Training feature dataset.
            y_train (Series): Training target dataset.
            x_test (DataFrame): Test feature dataset.
            n_splits (int): Number of splits for the K-Fold cross-validation.
        
        Returns:
            tuple: A tuple containing reshaped arrays of out-of-fold predictions for the training data and averaged predictions for the test data.
        """
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_train = np.zeros((ntrain,))
        oof_test_skf = np.empty((n_splits, ntest))

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            x_tr = x_train.iloc[train_index]
            y_tr = y_train.iloc[train_index]
            x_te = x_train.iloc[test_index]

            clone_model = model(**params)
            clone_model.fit(x_tr, y_tr)

            oof_train[test_index] = clone_model.predict(x_te)
            oof_test_skf[i, :] = clone_model.predict(x_test)

        oof_test = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    def get_oof_list(self, x_train, y_train, x_test):
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        num_models = len(self.base_models)
        self.oof_train = np.zeros((ntrain, num_models))
        self.oof_test = np.zeros((ntest, num_models))
    
        try:
            for idx, (model, params) in enumerate(self.base_models):
                oof_train, oof_test = self.get_oof(model, params, x_train, y_train, x_test, n_splits=5)
                self.oof_train[:, idx] = oof_train.ravel()
                self.oof_test[:, idx] = oof_test.ravel()
            return True  # Indicate successful completion
        except Exception as e:
            print(f"An error occurred: {e}")
            return False  # Indicate that an error occurred

    def meta_fit(self,meta_x_train, y_train):
        """
        Fit the meta-model using the prepared meta training dataset and the provided target values.
        
        Parameters:
            y_train (Series): The target values corresponding to the training dataset.
        """
        self.meta_model.fit(meta_x_train, y_train)  # Make sure y_train is accessible

    def predict(self, X):
        """
        Predict using the meta-model on the provided dataset.
        
        Parameters:
            X (DataFrame): The dataset to make predictions on.
        
        Returns:
            numpy.ndarray: The predicted values."""
        
        return self.meta_model.predict(X)

    def get_feature_importance(self):
        """
        Retrieve and aggregate feature importance from base models, if available.
        
        Returns:
            DataFrame: A DataFrame containing the feature importances from each base model.
        """
        feature_importance = pd.DataFrame()
        for i, (model, _) in enumerate(self.base_models):
            if hasattr(model, 'feature_importances_'):
                feature_importance[f'Model_{i}'] = model.feature_importances_
        return feature_importance
    
    def metric_evaluation(self,metric,y_pred,y_test):
        """
        Evaluate the model using the specified metric.
        
        Parameters:
            metric (function): A function that evaluates the prediction error.
            y_pred (numpy.ndarray): Predicted values.
            y_test (Series): Actual target values.
        
        Returns:
            float: The calculated metric score.
        """
        return metric(y_test, y_pred)

    def plot_feature_importances(self):
        """
        Plot the feature importances of the base models.
        """
        
    def plot_feature_importance_corr(self):
        """
        Plot correlations of feature importances between base models.
        """
        pass