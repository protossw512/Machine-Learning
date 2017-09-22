# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:05:27 2016

@author: hello
"""
from __future__ import division
import numpy
import random
from numpy import genfromtxt,append
from sklearn import svm
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

numpy.random.seed(123)
random.seed(789)

def SVM(x_train, y_train, x_test, y_test):
    clf = svm.SVC(C=3.5, cache_size=200, class_weight=None, coef0=0.7, 
        decision_function_shape=None, degree=3, gamma=0.3,
        kernel='poly', max_iter=-1, probability=False, 
        random_state=None, shrinking=True, tol=0.001, verbose=False)

#for 5000
#    clf = svm.SVC(C=3.5, cache_size=200, class_weight=None, coef0=0.7, 
#        decision_function_shape=None, degree=3, gamma=0.3,
#        kernel='poly', max_iter=-1, probability=False, 
#        random_state=None, shrinking=True, tol=0.001, verbose=False)

#for 10000
#    clf = svm.SVC(C=6, cache_size=200, class_weight=None, coef0=0.6, 
#        decision_function_shape=None, degree=3, gamma=0.3,
#        kernel='poly', max_iter=-1, probability=False, 
#        random_state=None, shrinking=True, tol=0.001, verbose=False)

    clf.fit(x_train, y_train)  
    
    #this part is for cross validation
    x_all = append(x_train, x_test, axis=0)
    y_all = append(y_train, y_test, axis=0)
    clf2 = svm.SVC(C=3.5, cache_size=200, class_weight=None, coef0=0.7, 
        decision_function_shape=None, degree=3, gamma=0.3,
        kernel='poly', max_iter=-1, probability=False, 
        random_state=None, shrinking=True, tol=0.001, verbose=False)
    scores = cross_val_score(clf2, x_all, y_all, cv=10)
    cross_acc = numpy.average(scores)
    
    test_prediction = clf.predict(x_test)
    test_score = clf.decision_function(x_test)
    tarin_prediction = clf.predict(x_train)
    test_acc = 0
    train_acc = 0
    for i in range(len(y_test)):
        if test_prediction[i] == y_test[i]:
            test_acc += 1
    test_acc = test_acc / len(x_test)
    for i in range(len(y_train)):
        if tarin_prediction[i] == y_train[i]:
            train_acc += 1
    train_acc = train_acc / len(x_train)

    roc = roc_auc_score(y_test, test_score)
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, test_score)
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='SVM(area = %0.2f)' % roc, lw = 2)
    plt.savefig('roc', dpi = 1000)
    return test_acc, train_acc, cross_acc, roc

def RandomForest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=5)
    clf.fit(x_train, y_train)

#for 10000
#    clf = RandomForestClassifier(n_estimators=15, criterion='entropy', max_depth=5)
#    clf.fit(x_train, y_train)
#for 5000
#    clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=5)

    
    #this part is for cross validation
    x_all = append(x_train, x_test, axis=0)
    y_all = append(y_train, y_test, axis=0)
    clf2 = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=5)
    scores = cross_val_score(clf2, x_all, y_all, cv=10)
    cross_acc = numpy.average(scores)
    
    test_prediction = clf.predict(x_test)
    positive = numpy.array(clf.predict_proba(x_test))
    test_score = numpy.delete(positive, 0, axis = 1)
    tarin_prediction = clf.predict(x_train)
    test_acc = 0
    train_acc = 0
    for i in range(len(y_test)):
        if test_prediction[i] == y_test[i]:
            test_acc += 1
    test_acc = test_acc / len(x_test)
    for i in range(len(y_train)):
        if tarin_prediction[i] == y_train[i]:
            train_acc += 1
    train_acc = train_acc / len(x_train)

    roc = roc_auc_score(y_test, test_score)
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, test_score)
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RF(area = %0.2f)' % roc, lw = 2)
    plt.savefig('roc', dpi = 1000)
    return test_acc, train_acc, cross_acc, roc

def NN(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(activation='relu', alpha=0.3, batch_size=30,
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(6, 6, 6), learning_rate='constant',
       learning_rate_init=0.001, max_iter=20000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, shuffle=True,
       solver='lbfgs', tol=0.00001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    clf.fit(x_train, y_train)

#for 5000
#    clf = MLPClassifier(activation='relu', alpha=0.3, batch_size=30,
#       beta_1=0.9, beta_2=0.999, early_stopping=False,
#       epsilon=1e-08, hidden_layer_sizes=(6, 6, 6), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=20000, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, shuffle=True,
#       solver='lbfgs', tol=0.00001, validation_fraction=0.1, verbose=False,
#       warm_start=False)
#for 10000
#    clf = MLPClassifier(activation='relu', alpha=0.3, batch_size=30,
#       beta_1=0.9, beta_2=0.999, early_stopping=False,
#       epsilon=1e-08, hidden_layer_sizes=(10, 10), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=20000, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, shuffle=True,
#       solver='lbfgs', tol=0.00001, validation_fraction=0.1, verbose=False,
#       warm_start=False)    
    #this part is for cross validation
    x_all = append(x_train, x_test, axis=0)
    y_all = append(y_train, y_test, axis=0)
    clf2 = MLPClassifier(activation='relu', alpha=0.3, batch_size=30,
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(6, 6, 6), learning_rate='constant',
       learning_rate_init=0.001, max_iter=20000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, shuffle=True,
       solver='lbfgs', tol=0.00001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    scores = cross_val_score(clf2, x_all, y_all, cv=10)
    cross_acc = numpy.average(scores)
    
    test_prediction = clf.predict(x_test)
    positive = numpy.array(clf.predict_proba(x_test))
    test_score = numpy.delete(positive, 0, axis = 1)
    tarin_prediction = clf.predict(x_train)
    test_acc = 0
    train_acc = 0
    for i in range(len(y_test)):
        if test_prediction[i] == y_test[i]:
            test_acc += 1
    test_acc = test_acc / len(x_test)
    for i in range(len(y_train)):
        if tarin_prediction[i] == y_train[i]:
            train_acc += 1
    train_acc = train_acc / len(x_train)
    
    roc = roc_auc_score(y_test, test_score)
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, test_score)
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='NN(area = %0.2f)' % roc, lw = 2)
    plt.savefig('roc', dpi = 1000)
    return test_acc, train_acc, cross_acc, roc
    
def Logistic(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
        C=1.0, fit_intercept=True, intercept_scaling=1, 
        class_weight=None, random_state=None, solver='liblinear', 
        max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    clf.fit(x_train, y_train)
    
    #this part is for cross validation
    x_all = append(x_train, x_test, axis=0)
    y_all = append(y_train, y_test, axis=0)
    clf2 = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
        C=1.0, fit_intercept=True, intercept_scaling=1, 
        class_weight=None, random_state=None, solver='liblinear', 
        max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    scores = cross_val_score(clf2, x_all, y_all, cv=10)
    cross_acc = numpy.average(scores)
    
    test_prediction = clf.predict(x_test)
    test_score = clf.decision_function(x_test)
    tarin_prediction = clf.predict(x_train)
    test_acc = 0
    train_acc = 0
    for i in range(len(y_test)):
        if test_prediction[i] == y_test[i]:
            test_acc += 1
    test_acc = test_acc / len(x_test)
    for i in range(len(y_train)):
        if tarin_prediction[i] == y_train[i]:
            train_acc += 1
    train_acc = train_acc / len(x_train)
    
    roc = roc_auc_score(y_test, test_score)
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, test_score)
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='LR(area = %0.2f)' % roc, lw = 2)
    plt.savefig('roc', dpi = 1000)
    return test_acc, train_acc, cross_acc, roc
    
if __name__ == "__main__":

#load dataset
    train_ori = genfromtxt('train5000.csv', dtype=float, delimiter=',', skip_header=0)
    test_ori = genfromtxt('test5000.csv', dtype=float, delimiter=',', skip_header=0)
    
    new_train = numpy.delete(train_ori, 9, 1)
    new_test = numpy.delete(test_ori, 9, 1)
#    
    y_train = train_ori[:,-1]
    y_test = test_ori[:,-1]
    
    
#Normalize dataset

    mean = numpy.array(numpy.mean(new_train, axis = 0))
    std = numpy.array(numpy.std(new_train, axis = 0))
    norm_train = (new_train - mean) / std
    norm_test = (new_test - mean) / std
    
    x_train = numpy.c_[numpy.ones(329), norm_train]
    x_test = numpy.c_[numpy.ones(100), norm_test]
    
    x_all = append(x_train, x_test, axis=0)
    test_acc, train_acc , cross_acc, roc= SVM(x_train, y_train, x_test, y_test)
    print "SVM: Training Acc:", train_acc, "Testing ACC:", test_acc, "Crossing Acc:", cross_acc, 'ROC:', roc
    test_acc, train_acc , cross_acc, roc= RandomForest(x_train, y_train, x_test, y_test)    
    print "RandomForest: Training Acc:", train_acc, "Testing ACC:", test_acc, "Crossing Acc:", cross_acc, 'ROC:', roc
    test_acc, train_acc , cross_acc, roc= NN(x_train, y_train, x_test, y_test)    
    print "NN: Training Acc:", train_acc, "Testing ACC:", test_acc, "Crossing Acc:", cross_acc, 'ROC:', roc
    test_acc, train_acc , cross_acc, roc= Logistic(x_train, y_train, x_test, y_test)    
    print "Logistic: Training Acc:", train_acc, "Testing ACC:", test_acc, "Crossing Acc:", cross_acc, 'ROC:', roc
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc', dpi = 1000)
#    plt.show()