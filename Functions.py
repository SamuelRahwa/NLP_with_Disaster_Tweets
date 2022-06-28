import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

# Warnings Off
import warnings
warnings.filterwarnings('ignore')

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb

# Scalers
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler


# Categorical Create Dummies
from sklearn.preprocessing import OneHotEncoder

# Classification Models
def run_class_model(model, X_train, y_train, X_test, y_test):

    # fitting
    model.fit(X_train, y_train)

    # predictions
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    
    # Classification Reports
    print('********************************************************\n')
    print('\033[1m' +'     Classification Report: Train\n' +'\033[0m')
    print(classification_report(y_train, y_hat_train))
    print('********************************************************\n')
    print('\033[1m' +'     Classification Report: Test\n' +'\033[0m')
    print(classification_report(y_test, y_hat_test))
    print('********************************************************\n')
    
    # Confusion Matrices
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 6))

    plot_confusion_matrix(model, X_train, y_train, ax=ax0,cmap=plt.cm.Blues);
    plot_confusion_matrix(model, X_test, y_test, ax=ax1,cmap=plt.cm.Greens);

    ax0.title.set_text('Train Confusion Matrix');
    ax1.title.set_text('Test Confusion Matrix');
    
    return model # return the model object