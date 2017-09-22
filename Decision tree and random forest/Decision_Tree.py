#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:08:02 2016

@author: Rasadell
"""
from __future__ import division
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline


random.seed(10)


#Devide dataset into two groups by threshold Theta
def get_sets(data, col, val):
    def _separation(line):
        if isinstance(val, float):
            return line[col] >= val
        else:
            return line[col] == val
    set1 = [line for line in data if _separation(line)]
    set2 = [line for line in data if not _separation(line)]
    return (set1, set2)

#Count how many unique labels in a dataset 
def labelcount(data):
    labels = {}
    for line in data:
        r = line[-1]
        if r not in labels:
            labels[r] = 0
        labels[r] += 1
    return labels

#Information gain calculation
def IG(data, entropy = 0.0):
    labels = labelcount(data)
    for cata in labels.keys():
        p = labels[cata] / len(data)
        entropy = entropy - p * math.log(p, 2)
    return entropy

#Define a tree class to store the decision tree. It is equal a data structure that used to store a tree
class tree(object):
    def __init__(self, col = - 1, val = None, label = None, left = None, right = None):
        self.col = col
        self.val = val
        self.label = label
        self.left = left
        self.right = right

#build a decision tree.
def buildtree(data, k, depth=0):
    if data == []:
        return tree()
    # Add stop condition
    if sum(labelcount(data).values()) <= k:
        max_key = max(labelcount(data), key = lambda k: labelcount(data)[k])
        return tree(label = {max_key: labelcount(data)[max_key]})
#        return tree(label = labelcount(data))
    current_gain = 0
    current_decision = None
    current_set = None
    
    for col in range(0, len(data[0]) - 1):
        column_values={}
        for line in data:
            column_values[line[col]]=1 
        for val in column_values.keys():
            (set1,set2)=get_sets(data,col,val)
            p=len(set1)/len(data)
            gain=IG(data)-p*IG(set1)-(1-p)*IG(set2)
            if depth == 0:
                print col,",",val,",",gain
            if gain>current_gain and len(set1)>0 and len(set2)>0:
                current_gain=gain
                current_decision=(col,val)
                current_set=(set1,set2)
                
    if current_gain>0:
        left=buildtree(current_set[0], k, depth+1)
        right=buildtree(current_set[1], k, depth+1)
        return tree(col=current_decision[0],val=current_decision[1],left=left,right=right)
    else:
        return tree(label=labelcount(data))

#Build random tree for random forest
def buildrandomtree(data, k):
    if data == []:
        return tree()
    # Add stop condition
    if sum(labelcount(data).values()) <= k:
        max_key = max(labelcount(data), key = lambda k: labelcount(data)[k])
        return tree(label = {max_key: labelcount(data)[max_key]})
#        return tree(label = labelcount(data))
    current_gain = 0
    current_decision = None
    current_set = None
    
    for col in random.sample([0,1,2,3],2): #Select 2 features from 4 randomly
        column_values={}
        for line in data:
            column_values[line[col]]=1 
        for val in column_values.keys():
            (set1,set2)=get_sets(data,col,val)
            p=len(set1)/len(data)
            gain=IG(data)-p*IG(set1)-(1-p)*IG(set2)
            if gain>current_gain and len(set1)>0 and len(set2)>0:
                current_gain=gain
                current_decision=(col,val)
                current_set=(set1,set2)
                
    if current_gain>0:
        left=buildrandomtree(current_set[0], k)
        right=buildrandomtree(current_set[1], k)
        return tree(col=current_decision[0],val=current_decision[1],left=left,right=right)
    else:
        return tree(label=labelcount(data))

#Random samples
def randomSample(data):
    new_data = []
    for i in range(0, len(data)):
        l = random.randint(0, len(data)-1)
        new_data.append(data[l])
    return new_data

#Print the tree struture.        
def printtree(tree,space = ""):
   # Is this a leaf node?
    if tree.label!=None:
        print str(tree.label)
    else:
        print str(tree.col)+':'+str(tree.val)+'? '
        # Print the branches
        print space + 'T->', printtree(tree.left, space + " ")
        print space + 'F->', printtree(tree.right, space + " ")

#Apply a new test data to tree model to get the prediction.
def classify(data,tree):
    def _classify(data,tree):
        if tree.label!=None:
            return tree.label.keys()
        else:
            v=data[tree.col]
            branch=None
        if v>=tree.val: branch=tree.left
        else: branch=tree.right
        return _classify(data,branch)
    predict = []
    for line in data:
        predict.append(_classify(line,tree))
    return predict

#Calculate prediction accuracy.        
def accuracy(test, predict):
    t = 0
    for n in range(len(test)):
        if test[n] == predict[n]:
            t += 1
    return t / len(test)

#Build a random forest and return test and training accuracy as a set.
def randomforest(train, train2, train_label, test, test_label, L, k):
    predicgroup_test = {}
    predicgroup_train = {}
    pred_test = []
    pred_train = []
    for i in range(0, L):
        tree = buildrandomtree(train, k)
        predicgroup_test[i] = classify(test, tree)
        predicgroup_train[i] = classify(train2, tree)

    for n in range(0, len(test)):
        result = {}
        for i in range(0, L):
            if predicgroup_test[i][n][0] not in result:
                result[predicgroup_test[i][n][0]] = 1
            result[predicgroup_test[i][n][0]] += 1
        max_key = max(result, key = lambda k: result[k])
        pred_test.append([max_key])

    for n in range(0, len(train2)):
        result = {}
        for i in range(0, L):
            if predicgroup_train[i][n][0] not in result:
                result[predicgroup_train[i][n][0]] = 1
            result[predicgroup_train[i][n][0]] += 1
        max_key = max(result, key = lambda k: result[k])
        pred_train.append([max_key])
    return accuracy(test_label, pred_test), accuracy(train_label, pred_train)

#Plot figures
def plot_fig(x_label, y_label):
    new1_x = np.linspace(min(plot1_x), max(plot1_x), 1000)
    new1_y = spline(plot1_x, plot1_y, new1_x)
    new2_x = np.linspace(min(plot2_x), max(plot2_x), 1000)
    new2_y = spline(plot2_x, plot2_y, new2_x)
    plt.plot(new1_x, new1_y, 'b', label='Test Accuracy')
    plt.plot(new2_x, new2_y, 'g', label='Train Accuracy')
    # plt.plot(plot1_x, plot1_y, 'b', label='Test Accuracy')
    # plt.plot(plot2_x, plot2_y, 'g', label='Train Accuracy')
    plt.legend(loc=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
if __name__ == "__main__":

#Load training data
    ori_list = []
    train_list = []
    with open('iris_train.csv', 'rb') as f:
        reader = csv.reader(f)
        ori_list = list(reader)
    for r in ori_list:
        l = map(float, r[0].split(';'))
        train_list.append(l) 
    train_list = [[line[0], line[1], line[2], line[3], int(line[4])] for line in train_list]
    train_list2 = [[line[0], line[1], line[2], line[3]] for line in train_list]
    train_label = [[line[4]] for line in train_list]
#Load testing data
    ori_list = []
    test_list = []
    with open('iris_test.csv', 'rb') as f:
        reader = csv.reader(f)
        ori_list = list(reader)
    for r in ori_list:
        l = map(float, r[0].split(';'))
        test_list.append(l)
    test_label = [[int(line[4])] for line in test_list]
    test_list = [[line[0], line[1], line[2], line[3]] for line in test_list]

#Product an array to store accuracy under different k values (part I)
#    test_acc = []
#    train_acc = []
#    plot1_x = []
#    plot2_x = []
#    for k in range(1, 120, 2):
#        plot1_x.append(k)
#        plot2_x.append(k)
#        t = buildtree(train_list, k)
#        predict_test = classify(test_list, t)
#        test_acc.append(accuracy(test_label, predict_test))
#        predict_train = classify(train_list2, t)
#        train_acc.append(accuracy(train_label, predict_train))
#    plot1_y = test_acc
#    plot2_y = train_acc
#    plot_fig("K","Accuracy")

#    t = buildtree(train_list, 1)

#product an array to store accuracy under different L values(part II)
#    plot1_x = []
#    plot2_x = []
#    plot1_y = []
#    plot2_y = []
#    for L in range(5,35, 5):
#        for k in range(1, 120, 5):
#            test_acc = []
#            train_acc = []
#            plot1_x.append(k)
#            plot2_x.append(k)
#            for i in xrange(10):
#                random_list = randomSample(train_list)
#                random_label = [[int(line[4])] for line in random_list]
#                testacc, trainacc = randomforest(random_list, random_list, random_label, test_list, test_label, L, k)
#                test_acc.append(testacc)
#                train_acc.append(trainacc)
#            avg1 = sum(test_acc)/len(test_acc)
#            plot1_y.append(avg1)
#            avg2 = sum(train_acc)/len(train_acc)
#            plot2_y.append(avg2)
#        plot_fig('K', 'Accuracy')

#Plot testing/training accuracy on different L values given a fixed k(part III)
    plot1_x = []
    plot2_x = []
    plot1_y = []
    plot2_y = []
    for L in range(1,35, 5):
        test_acc = []
        train_acc = []
        plot1_x.append(L)
        plot2_x.append(L)
        for i in xrange(10):
            random_list = randomSample(train_list)
            random_label = [[int(line[4])] for line in random_list]
            testacc, trainacc = randomforest(random_list, random_list, random_label, test_list, test_label, L, 80)
            test_acc.append(testacc)
            train_acc.append(trainacc)
        avg1 = sum(test_acc)/len(test_acc)
        plot1_y.append(avg1)
        avg2 = sum(train_acc)/len(train_acc)
        plot2_y.append(avg2)
    plot_fig('L', 'Accuracy')
        

        

    
    