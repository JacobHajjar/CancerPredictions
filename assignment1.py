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

''' develop the best predictive model based on the chemical engineering dataset'''

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


    #split trainingand testing
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.25, random_state = 0)
    
    #create models from data
    log, svc_lin, knn, forest = model_data(x_data, y_data)

    #create pipeline to scale data, then run model
    sc = StandardScaler()
    log_pipe = make_pipeline(sc, log)
    svc_lin_pipe = make_pipeline(sc, svc_lin)
    knn_pipe = make_pipeline(sc, knn)
    forest_pipe = make_pipeline(sc, forest)

    #create cross validation split
    #cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=0)

    #train and create cross validated prediction
    #log_predictions = log.predict(x_test)
    log_predictions = cross_val_predict(log_pipe, x_data, y_data, cv=2)
    #svc_lin_predictions = svc_lin.predict(x_test)
    svc_lin_predictions = cross_val_predict(svc_lin_pipe, x_data, y_data, cv=2)
    #knn_predictions = knn.predict(x_test)
    knn_predictions = cross_val_predict(knn_pipe, x_data, y_data, cv=2)
    #forest_predictions = forest.predict(x_test)
    forest_predictions = cross_val_predict(forest_pipe, x_data, y_data, cv=2)
    
    #print(cross_val_score(log_pipe, x_data, y_data, cv=cv))

    #print_cm_classification_report("Logistic Regression", y_test, log_predictions)
    print_cm_classification_report("Logistic Regression", y_data, log_predictions)
    #print_cm_classification_report("Support Vector Machine Classifier", y_test, svc_lin_predictions)
    print_cm_classification_report("Support Vector Machine Classifier", y_data, svc_lin_predictions)
    #print_cm_classification_report("K-nearest neighbors", y_test, knn_predictions)
    print_cm_classification_report("K-nearest neighbors", y_data, knn_predictions)
    #print_cm_classification_report("Forest Classifier", y_test, forest_predictions)
    print_cm_classification_report("Forest Classifier", y_data, forest_predictions)
    plt.legend()

    '''[["Age","Number of sexual partners","First sexual intercourse",
    "Num of pregnancies","Smokes (years)","Smokes (packs/year)", "Hormonal Contraceptives (years)",
    "IUD","IUD (years)","STDs","STDs (number)","STDs:condylomatosis","STDs:cervical condylomatosis","STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis","STDs:syphilis","STDs:pelvic inflammatory disease","STDs:genital herpes","STDs:molluscum contagiosum",
    "STDs:AIDS","STDs:HIV","STDs:Hepatitis B","Dx:HPV","STDs: Number of diagnosis"]]
    current_feature = data_frame[["Smokes (years)"]]
    log.fit(current_feature / np.std(current_feature, 0), y_data)
    print(log.coef_)'''

    #standardize data
    sc = StandardScaler()
    x_data_scaled = sc.fit_transform(x_data)
    log.fit(x_data_scaled, y_data)
    coefs = pd.DataFrame(log.coef_.T, columns=['Coefficients'], index=x_data.columns)
    coefs.plot(kind='barh', figsize=(9, 7))
    plt.title('Ridge model')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)
    plt.show()
    
    #print(y_test)
    #print(pred_prob)
    #for y_index, y in enumerate(y_test):
        #print(y, pred_prob[y_index][1])
        #print(predictions[y_index])
        #print(pred_prob[y_index][1])
    #Another way to get the models accuracy on the test data


def plot_ROC_curve(model_name, y_test, predictions):
    fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
    auc = round(metrics.roc_auc_score(y_test, predictions), 4)
    plt.plot(fpr,tpr,label=model_name+", AUC="+str(auc))

def print_cm_classification_report(model_name, y_test, predictions):

    print(model_name)
    cm = confusion_matrix(y_test, predictions)
    
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    
    print(cm)
    #print('Model Testing Accuracy = "{}"'.format((TP + TN) / (TP + TN + FN + FP)))
    #print( accuracy_score(y_test, chosen_model.predict(x_test)))
    print()# Print a new line
    #Check precision, recall, f1-score
    print( classification_report(y_test, predictions, zero_division = 0))
    #plot_ROC_curve(model_name, y_test, predictions)

def model_data(X_train, Y_train):

    #Using Logistic Regression 
    log = LogisticRegression(random_state = 0, n_jobs=8)
    #log.fit(X_train, Y_train)

    #Using SVC linear
    svc_lin = SVC(kernel = 'linear', random_state = 0, probability=True)
    #svc_lin.fit(X_train, Y_train)

    #Using KNeighborsClassifier 
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2, n_jobs=8)
    #knn.fit(X_train, Y_train)

    #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0, n_jobs=8)
    #forest.fit(X_train, Y_train)

    #print model accuracy on the training data.
    #print('Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    #print('Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    #print('Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
    #print('K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    return log, svc_lin, knn, forest

if __name__ == '__main__':
    main()
