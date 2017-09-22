import numpy
import random
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy.interpolate import spline

numpy.random.seed(123)
random.seed(789)

## Gradient descent method for L2 function 
def learn (Length, Lambda, x_train, x_test, y_train, y_test):
    w = numpy.random.randn(num_fea) 
    for i in range (0, 1000):
        y_pred = numpy.dot(x_train,w)
        sqrerr = (y_pred - y_train)*(y_pred - y_train)
        loss = sqrerr.sum()
        y_test_pred = numpy.dot(x_test,w)
        test_sqrerr = (y_test_pred - y_test)*(y_test_pred - y_test)
        testing_loss = test_sqrerr.sum()
        gradient = 2 * numpy.dot((y_pred - y_train),x_train) + 2 * Lambda*w
        w = w - Length*gradient
        i += 1
    print "Learning Rate:", Length, "Lambda", Lambda, "Training Loss:",loss, "Testing Loss:", testing_loss, "|gradient|", numpy.linalg.norm (gradient)
    plot1_x.append(Length)
    plot1_y.append(loss)

## Function for exploring different Lambda value using closed form solution
def closed_form_learn (Length, Lambda, x_train, x_test, y_train, y_test):
    w = closed_form(Lambda, x_train, y_train) 
    y_pred = numpy.dot(x_train,w)
    sqrerr = (y_pred - y_train)*(y_pred - y_train)
    loss = sqrerr.sum()
    y_test_pred = numpy.dot(x_test,w)
    test_sqrerr = (y_test_pred - y_test)*(y_test_pred - y_test)
    testing_loss = test_sqrerr.sum()
    plot1_x.append(math.log(Lambda))
    plot1_y.append(testing_loss)
    print "Learning Rate:", Length, "Lambda", Lambda, "Training Loss:",loss, "Testing Loss:", testing_loss    

## Cross validation function    
def cross_val(Length, Lambda, x_train, y_train):
    sum_SSE = 0
    for i in range(10):
        begin = 10*i
        end   = 10*i+10
        tmp_testX = x_train[begin:end,:]
        tmp_testY = y_train[begin:end]
        tmp_trainX = numpy.delete(x_train, numpy.s_[begin:end], 0)
        tmp_trainY = numpy.delete(y_train, numpy.s_[begin:end], 0)             
        w = closed_form(Lambda, tmp_trainX, tmp_trainY)
        y_test_pred = numpy.dot(tmp_testX,w)
        test_sqrerr = (y_test_pred - tmp_testY)*(y_test_pred - tmp_testY)
        testing_loss = test_sqrerr.sum()
        sum_SSE += testing_loss
    plot1_x.append(math.log(Lambda))
    plot1_y.append(sum_SSE)
    print "Lambda:", Lambda, "sum_SSE:", sum_SSE


## Calculate closed form solution for L2 linear regression loss function         
def closed_form (Lambda, x_train, y_train):
    I = numpy.identity(45)
    w = numpy.linalg.inv(Lambda*I + numpy.dot(numpy.transpose(x_train), x_train))
    w = numpy.dot(w, numpy.transpose(x_train))
    w = numpy.dot(w, y_train)
    return w

## Plot function
def plot_fig(x_label, y_label):
    new1_x = numpy.linspace(min(plot1_x), max(plot1_x), 1000)
    new1_y = spline(plot1_x, plot1_y, new1_x)
    plt.plot(new1_x, new1_y, 'b')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
if __name__ == "__main__":

    ## Load dataset
    train_ori = genfromtxt('train p1-16.csv', dtype=float, delimiter=',', skip_header=0)
    test_ori = genfromtxt('test p1-16.csv', dtype=float, delimiter=',', skip_header=0)
    
    new_train = numpy.delete(train_ori, [0,45], 1)
    new_test = numpy.delete(test_ori, [0,45], 1)
    
    ## Preprocess & Normalize dataset    
    y_train = train_ori[:,-1]
    y_test = test_ori[:,-1]


    mean = numpy.array(numpy.mean(new_train, axis = 0))
    std = numpy.array(numpy.std(new_train, axis = 0))
    norm_train = (new_train - mean) / std
    norm_test = (new_test - mean) / std

    ## Add dummy features back to training and testing dataset    
    x_train = numpy.c_[numpy.ones(100), norm_train]
    x_test = numpy.c_[numpy.ones(100), norm_test]

    ## Number of features
    num_fea = 45
    
    ## Testing different paramaters
    Lambdas = [0.00035, 0.01, 0.03, 0.1, 0.3, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 10, 100]
    #Lengths = [0.00001, 0.0001, 0.0002, 0.0003, 0.0004,0.0005, 0.0006, 0.0007, 0.0008]
    #Lengths = [0.5, 0.1, 0.05 , 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    Lengths = [0.00001, 0.00005, 0.00007, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]
    
    plot1_x = []
    plot1_y = []

    ## Testing different learning rate (part 1)
    for i in Lengths:
        learn(i, 1, x_train, x_test, y_train, y_test)
    ## Testing different Lambdas using closed form solution (part 2)
    #for i in Lambdas:
        #closed_form_learn(0.0005, i, x_train, x_test, y_train, y_test)
    ## Testing different Lambdas using cross-validation (part 3)
    #for i in Lambdas:
        #cross_val(0.0005, i, x_train, y_train)
    plot_fig("Learning Rate","Loss")
