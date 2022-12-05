#!/usr/bin/env python3
from pydoc import doc
import sys
import time
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt 
import seaborn as sns

''' develop the best predictive model based on the cervical cancer dataset'''

__author__ = 'Jacob Hajjar'
__email__ = 'hajjarj@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar'


def main():
    """the main function"""
    data_frame = pd.read_csv("clean_risk_factors_cervical_cancer.csv")
    #get relevant data
    x_data = data_frame[["Age","Number of sexual partners","First sexual intercourse",
    "Num of pregnancies","Smokes (years)","Smokes (packs/year)", "Hormonal Contraceptives (years)",
    "IUD","IUD (years)","STDs","STDs (number)","STDs:condylomatosis","STDs:cervical condylomatosis","STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis","STDs:syphilis","STDs:pelvic inflammatory disease","STDs:genital herpes","STDs:molluscum contagiosum",
    "STDs:AIDS","STDs:HIV","STDs:Hepatitis B","Dx:HPV","STDs: Number of diagnosis"]]

    y_data = data_frame["Dx:Cancer"]

    #create models from data
    sc = StandardScaler()
    x_scaled = sc.fit_transform(x_data)
    log = model_data(x_scaled, y_data)

    prediction = log.predict_proba(x_scaled)
    pd_df = pd.DataFrame(prediction, columns = ['no_prob', 'cancer_prob'])
    mean = pd_df['cancer_prob'].sum() / len(prediction)

    example_data = [[25, 1, 18, 1, 5, 10, 3, 0, 0, 1, 1, 1, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    

    print(predict_risk(sc, log, mean, example_data))
    

def model_data(x_data, y_data):
    #Using Logistic Regression 
    log = LogisticRegression(random_state = 0, n_jobs=8).fit(x_data, y_data)
    return log

def predict_risk(scalar, model, mean, cancer_factors):
    '''accepts the scalar for the data, model, and a list of the x data in order, return risk 1, 2, 3 for low, medium, high'''
    """Age","Number of sexual partners","First sexual intercourse","Num of pregnancies","Smokes (years)","Smokes (packs/year)", 
    "Hormonal Contraceptives (years)","IUD","IUD (years)","STDs","STDs (number)","STDs:condylomatosis","STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis","STDs:vulvo-perineal condylomatosis","STDs:syphilis","STDs:pelvic inflammatory disease",
    "STDs:genital herpes","STDs:molluscum contagiosum","STDs:AIDS","STDs:HIV","STDs:Hepatitis B","Dx:HPV","STDs: Number of diagnosis"""
    x_scaled = scalar.transform(cancer_factors)
    cancer_prob = model.predict_proba(x_scaled)
    risk_level = 1
    if cancer_prob > mean:
        risk_level = 2
    if cancer_prob > 0.5:
        risk_level = 3
    return risk_level


if __name__ == '__main__':
    main()
