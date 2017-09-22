# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:52:47 2016

@author: hello
"""
from __future__ import division
import random
import numpy as np
from numpy import dot, mean, argsort
from numpy.linalg import eigh

#random.seed(1024)

#File reading function
def readin(filename):
    return [x.strip() for x in open(filename, "r").readlines()]

#convert strings to list of numbers
def convert_to_list(ori_list):
    final_list=[]
    for row in ori_list:
        l = map(float, row.split())
        final_list.append(l)
    return final_list

#convert labels to int
def convert_to_int(ori_list):
    final_list = []
    for e in ori_list:
        if isinstance(e, str):
            final_list.append(int(e))
    return final_list

#given a training data, apply kmean alrogrithm and out put prediction labels
def kmean(train):
    class0, class1 = random.sample(range(len(train)), 2)
    center0 = np.array(train[class0])
    center1 = np.array(train[class1])
    pred = [0] * len(train)
    change = 1
    while change > 0:
        change = 0
        total0 = np.array([0.0]*(train.shape)[1])
        total1 = np.array([0.0]*(train.shape)[1])
        for i in range(len(train)):
            vector = np.array(train[i])
            dist0 = np.linalg.norm(vector - center0)
            dist1 = np.linalg.norm(vector - center1)
            temp = 0
            if dist1 < dist0:
                temp = 1
                total1 += train[i]
            else:
                total0 += train[i]
            if temp != pred[i]:
                change += 1 
            pred[i] = temp
        center0 = total0 / (len(pred) - sum(pred))
        center1 = total1 / sum(pred)
    return pred

#given prediction from kmean and ground trouth, output mistakes
def SEE(pred, data, test_label):
    n = 0
    cluster0 = []
    cluster1 = []
    for i in range(len(pred)):
        if pred[i] == 0:
            cluster0.append(test_label[i])
        else:
            cluster1.append(test_label[i])
    if sum(cluster0) / len(cluster0) > 0.5:
        n += sum(cluster0) + len(cluster1) - sum(cluster1)
    else:
        n += len(cluster0) - sum(cluster0) + sum(cluster1) 
    return n
#given mistakes, calculate purity
def purity(data,n):
    pre = []
    for _ in range(10):    
        prediction = kmean(data[:,:n])
        pre.append(SEE(prediction, data, walking_label)/len(walking_label))
    return max(pre)        

#PCA algorithm, given a data, return projected data and normalized eigenvalues(for percentage calculation)
def PCA(data):
#shift data to mean=0
    data -= mean(data, 0)
    N = data.shape[1]
    C = np.zeros((N,N))
#calculate corvariance matrix
    for j in range(N):
        C[j, j] = mean(data[:, j] * data[:, j])
        for k in range(N):
            C[j, k] = C[k, j] = mean(data[:, j] * data[:, k])
#eigenvalues and eeigenmatrix
    E, V = eigh(C)
    key = argsort(E)[::-1]
    E, V = E[key], V[:, key]
    W = dot(V.T, data.T).T
    return W, E / (sum(E))

#calculate percentage for Q2
def percentage(E, p):
    total = 0
    n = 0
    for k in E:
        if total < p:
            total += k
            n += 1
        else:
            return n 

#LDA method for calculating vctor w
def LDA(data, label):
    C0 = []
    C1 = []
    N = data.shape[1]
    for i in range(len(label)):
        if label[i] == 1:
            C1.append(data[i])
        else:
            C0.append(data[i])
#c0, c1: data for each two classes
    c0 = np.array(C0)
    c1 = np.array(C1)
#m0, m1: mean vector of two classes
    m0 = np.mean(c0, 0)
    m0 = np.array([m0])
    m1 = np.mean(c1, 0)
    m1 = np.array([m1])
#initialize within class scatter matrix for two classes
    s0 = np.zeros((N,N))
    s1 = np.zeros((N,N))
#calculate s1 and s2
    for i in range(len(c0)):
        tran = np.array([c0[i]])
        s0 += (tran.T - m0.T).dot((tran.T - m0.T).T)
    for i in range(len(c1)):
        tran = np.array([c1[i]])
        s1 += (tran.T - m1.T).dot((tran.T - m1.T).T)   
#s is Sw        
    s = s0 + s1
    for i in range(N):
        s[i,i] += 1e-6
#calculate inverse
    S = np.linalg.inv(s)
#dot products to calculate w
    w = dot(S,(m0 - m1).T)
#W is the projection vector of our data onto w
    W = dot(w.T, walking_data.T).T.reshape((2059,1))
    return W

if __name__ == "__main__":
#loading data
    ori_walking_data = readin("walking.train.data")
    ori_walking_label = readin("walking.train.labels")
    walking_data = np.array(convert_to_list(ori_walking_data))
    walking_label = np.array(convert_to_int(ori_walking_label))
    W = LDA(walking_data,walking_label)

#Q2
    new_walking_data, E = PCA(walking_data)
    p_80 = percentage(E, 0.8)
    p_90 = percentage(E, 0.9)
#use sklearn packeg to testify the LDA method
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    train = lda.fit(walking_data, walking_label).transform(walking_data)
#print result for Q1, Q2 and Q3
    print 'Purity of original data: ', purity(walking_data, 477)
    print 'Purity of 1 PCs: ', purity(new_walking_data, 1)
    print 'Purity of 2 PCs: ', purity(new_walking_data, 2)
    print 'Purity of 3 PCs: ', purity(new_walking_data, 3)
    print 'Dimention left for 80% vairance: ', p_80
    print 'Dimention left for 90% variance: ', p_90
    print 'Purity of LDA method:', purity(W,1)    
    print 'sklearn package LDA purity: ', purity(train, 1)
    
    

    