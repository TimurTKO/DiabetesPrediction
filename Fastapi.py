import pandas as pd
from fastapi import FastAPI, UploadFile, File
import io
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.utils import compute_sample_weight
#
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import RobustScaler

from sklearn.cluster import KMeans

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from joblib import dump, load

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

from time import sleep

from sklearn.model_selection import cross_val_score

import optuna

from sklearn.metrics import fbeta_score, make_scorer

import mlflow

import json
from typing import List
from pydantic import BaseModel



app = FastAPI()

model=joblib.load('bestmodel.joblib')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content=await file.read()
    df=pd.read_csv(io.StringIO(content.decode('utf-8')))
    print(df.columns)
    predict_model=model.predict(df)
    print(type(predict_model))
    return {"predict": predict_model.tolist()}


