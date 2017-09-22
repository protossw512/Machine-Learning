from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
#from scipy.interpolate import spline


#Read txt file into lines of strings
def readin(filename):
    return [x.strip() for x in open(filename, "r").readlines()]

#Convert strings into integers
def convert_to_intlist(ori_list):
    final_list=[]
    for row in ori_list:
        l = map(int, row.split())
        final_list.append(l)
    return final_list

#Count how many times each word appeared in each sample using sparse matrix for training data   
def count_words(x_list, y_list, beta):
    if beta == 0:
        freq_mat = np.zeros(shape=(len(y_list)-int(y_list.sum()), dic_len))
    elif beta == 1:
        freq_mat = np.zeros(shape=(int(y_list.sum()), dic_len))
        
    mat_idx = 0
    for (x,y) in zip(xrange(len(x_list)), xrange(len(y_list))):
        if y_list[y] == beta:
            for it in x_list[x]:
                freq_mat[mat_idx][it] += 1
            mat_idx += 1
    
    return freq_mat

#Count how many times each word appeared in each sample for using sparse matrix for testing data    
def count_words_test(x_list):
    freq_mat = np.zeros(shape=(len(x_list), dic_len))
    
    for xrow in xrange(len(x_list)):
        for it in x_list[xrow]:
            freq_mat[xrow][it] += 1
    
    return freq_mat

#Calculate frequency of each words for a given freq_mat (multinomial model)
def sum_count_words_mult(freq_mat):
    sum_words = []
    for i in xrange(len(freq_mat[0])):
        sum_words.append(sum(freq_mat[:,i]))
    
    return sum_words
    
#Calculate frequency of each words for a given freq_mat (Bernoulli model)
def sum_count_words_bern(freq_mat):
    sum_words_01 = []
    for i in xrange(len(freq_mat[0])):
        sum_words_01.append(np.count_nonzero(freq_mat[:,i]))
    
    return sum_words_01

#Calculate probability of each words for a given freq_mat (Bernoulli model), used for calculating P(Xi | y = lambda)        
def prob_list_Bern(sum_words_01, y_train, beta):
    prob_Bern = []
    m = 0
    if beta == 0:
        m = len(y_train)-sum(y_train)
    elif beta == 1:
        m = sum(y_train)
    
    for i in xrange(len(sum_words_01)):
        prob_Bern.append((sum_words_01[i]+1)/(m+2))
    
    return prob_Bern
   
#Calculate probability of each words for a given freq_mat (multinomial model), used for calculating P(Xi | y = lambda)    
def prob_list_Mult(sum_words,  y_train, beta, alpha):
    prob_Mult = []
    m = 0
    if beta == 0:
        m = len(y_train)-sum(y_train)
    elif beta == 1:
        m = sum(y_train)
        
    v = len(sum_words)
    
    for i in xrange(len(sum_words)):
        prob_Mult.append((alpha+sum_words[i])/(m+(v*alpha)))
    
    return prob_Mult   

#Transfer categoary data into 0 and 1    
def trans_namelist(name_list):
    complete_namelist=np.zeros(len(name_list))
    for i in xrange(len(name_list)):
        complete_namelist[i] = trans_name(name_list[i])
    return complete_namelist

#Helper function for trans_namelist
def trans_name(name):
    if name == "HillaryClinton":
        return 0
    elif name == "realDonaldTrump":
        return 1    
        
#def plot_fig(x_label, y_label):
#    new1_x = np.linspace(min(plot1_x), max(plot1_x), 1000)
#    new1_y = spline(plot1_x, plot1_y, new1_x)
#    new2_x = np.linspace(min(plot2_x), max(plot2_x), 1000)
#    new2_y = spline(plot2_x, plot2_y, new2_x)
##    plt.plot(new1_x, new1_y, 'b')
#    plt.plot(new1_x, new1_y, 'b', label='Bernoulli')
#    plt.plot(new2_x, new2_y, 'g', label='Multinomial')
#    plt.legend(loc='upper right')
##    plt.plot(plot1_x, plot1_y, 'b')
#    plt.xlabel(x_label)
#    plt.ylabel(y_label)
#    plt.show()
    
#Use Bernoulli model to classify the testing data. 
def Bern_main(log_prob_t, log_one_prob_t, log_prob_h, log_one_prob_h, test_m, filename):
    beta_h = np.zeros(shape=(len(test_m), 1))
    beta_t = np.zeros(shape=(len(test_m), 1))
    for row in xrange(len(test_m)):
        for col in xrange(len(test_m[row])):
            if test_m[row][col] == 0:
                beta_h[row] += log_one_prob_h[col]
                beta_t[row] += log_one_prob_t[col]
            elif test_m[row][col] > 0:
                beta_h[row] += log_prob_h[col]
                beta_t[row] += log_prob_t[col]
            else:
                print "Error in testing data"
    
    sum_h = np.add(np.log(prob_y_h), beta_h)
    sum_t = np.add(np.log(prob_y_t), beta_t)
    
    new_sum_h = np.multiply(100, sum_h)
    new_sum_t = np.multiply(100, sum_t)

    res_mat_bern = []
    for row in xrange(len(new_sum_h)):
        if new_sum_h[row] >= new_sum_t[row]:
            res_mat_bern.append(0)
        else:
            res_mat_bern.append(1)
    
    res_mat_bern = np.array(res_mat_bern)
#    result = np.count_nonzero(res_mat_bern - y_test)
    
    # Calculate confusion matrix
#    P1 = np.count_nonzero(res_mat_bern)
#    P0 = len(res_mat_bern) - P1
#    T1 = np.count_nonzero(y_test)
#    T0 = len(y_test) - T1
#    p0t0 = 0
#    p0t1 = 0
#    p1t0 = 0
#    p1t1 = 0
#    for i in range(0, len(y_test)):
#        if res_mat_bern[i] == 0 and y_test[i] == 0:
#            p0t0 += 1
#        elif res_mat_bern[i] == 0 and y_test[i] == 1:
#            p0t1 += 1
#        elif res_mat_bern[i] == 1 and y_test[i] == 0:
#            p1t0 += 1
#        elif res_mat_bern[i] == 1 and y_test[i] == 1:
#            p1t1 += 1
#    print "Bern: ", "P1: ", P1, "P0: ", P0, "T1: ", T1, "T0: ", T0, "Total: ", len(y_test), \
#            "P0T0: ", p0t0, "P0T1: ", p0t1, "P1T0: ", p1t0, "P1T1: ", p1t1           
    # Import result
    res_bern_output = open(filename, "w")
    for i in res_mat_bern:
        if i == 0:
            res_bern_output.write("HillaryClinton" + "\n")
        else:
            res_bern_output.write("realDonaldTrump" + "\n")
    res_bern_output.close()
#    return result
    
#Use Multinomial model to classify the testing data. 
def Mult_main(log_prob_t, log_prob_h, test_m, filename):
    beta_h = np.zeros(shape=(len(test_m), 1))
    beta_t = np.zeros(shape=(len(test_m), 1))
    for row in xrange(len(test_m)):
        for col in xrange(len(test_m[row])):
            if test_m[row][col] > 0:
                beta_t[row] += test_m[row][col] * log_prob_t[col]
                beta_h[row] += test_m[row][col] * log_prob_h[col]
    
    sum_t = np.add(np.log(prob_y_t), beta_t)
    sum_h = np.add(np.log(prob_y_h), beta_h)
    
    new_sum_t = np.multiply(100, sum_t)
    new_sum_h = np.multiply(100, sum_h)

    res_mat_mult = []
    for row in xrange(len(new_sum_h)):
        if new_sum_h[row] >= new_sum_t[row]:
            res_mat_mult.append(0)
        else:
            res_mat_mult.append(1)
            
    res_mat_mult = np.array(res_mat_mult)
#    result = np.count_nonzero(res_mat_mult - y_test)

    # Calculate confusion matrix
#    P1 = np.count_nonzero(res_mat_mult)
#    P0 = len(res_mat_mult) - P1
#    T1 = np.count_nonzero(y_test)
#    T0 = len(y_test) - T1
#    p0t0 = 0
#    p0t1 = 0
#    p1t0 = 0
#    p1t1 = 0
#    for i in range(0, len(y_test)):
#        if res_mat_mult[i] == 0 and y_test[i] == 0:
#            p0t0 += 1
#        elif res_mat_mult[i] == 0 and y_test[i] == 1:
#            p0t1 += 1
#        elif res_mat_mult[i] == 1 and y_test[i] == 0:
#            p1t0 += 1
#        elif res_mat_mult[i] == 1 and y_test[i] == 1:
#            p1t1 += 1
#    print "Mult: ", "P1: ", P1, "P0: ", P0, "T1: ", T1, "T0: ", T0, "Total: ", len(y_test), \
#            "P0T0: ", p0t0, "P0T1: ", p0t1, "P1T0: ", p1t0, "P1T1: ", p1t1   
    # Import result
    res_mult_output = open(filename, "w")
    for i in res_mat_mult:
        if i == 0:
            res_mult_output.write("HillaryClinton" + "\n")
        else:
            res_mult_output.write("realDonaldTrump" + "\n")
    res_mult_output.close()
#    return result
    
def top_ten_words():
    sort_Bern_h = sorted(range(len(prob_list_bern_h)), key=prob_list_bern_h.__getitem__)
    sort_Bern_t = sorted(range(len(prob_list_bern_t)), key=prob_list_bern_t.__getitem__)
    sort_Mult_h = sorted(range(len(prob_list_mult_h)), key=prob_list_mult_h.__getitem__)
    sort_Mult_t = sorted(range(len(prob_list_mult_t)), key=prob_list_mult_t.__getitem__)
    
    topten_Bern_h = sort_Bern_h[-1:-11:-1]
    topten_Bern_t = sort_Bern_t[-1:-11:-1]
    topten_Mult_h = sort_Mult_h[-1:-11:-1]
    topten_Mult_t = sort_Mult_t[-1:-11:-1]

    top_ten_words_h = []
    top_ten_prob_h   = []
    top_ten_words_t = []
    top_ten_prob_t   = []

    top10_Mult_h = []
    top10_Mult_t = []
    top10_prob_Mult_h =[]
    top10_prob_Mult_t =[]
 
    for i in topten_Mult_h:
        top10_Mult_h.append(init_dic[i])
        top10_prob_Mult_h.append(prob_list_mult_h[i])
        
    for i in topten_Mult_t:
        top10_Mult_t.append(init_dic[i])
        top10_prob_Mult_t.append(prob_list_mult_t[i])

    for i in topten_Bern_h:
        top_ten_words_h.append(init_dic[i])
        top_ten_prob_h.append(prob_list_bern_h[i])
    
    for i in topten_Bern_t:
        top_ten_words_t.append(init_dic[i])
        top_ten_prob_t.append(prob_list_bern_t[i])
    print "Using Bern :"
    print "\n"
    print "top ten words for trump is " ,zip(top_ten_words_t, top_ten_prob_t)
    print "\n"
    print "top ten words for clinton is " ,zip(top_ten_words_h, top_ten_prob_h)
    print "\n\n"
    
    print "Using Mult :"
    print "\n"
    print "top ten words for trump is " ,zip(top10_Mult_t, top10_prob_Mult_t)
    print "\n"
    print "top ten words for clinton is " ,zip(top10_Mult_h, top10_prob_Mult_h)
    print "\n"

#def plot_different_alpha():
#    mult_alpha = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.5,1]
#    for it in mult_alpha:
#        prob_list_mult_t = prob_list_Mult(sum_words_mult_t, y_train, 1, it)
#        log_prob_t = np.log(prob_list_mult_t)
#        prob_list_mult_h = prob_list_Mult(sum_words_mult_h,y_train, 0, it)
#        log_prob_h = np.log(prob_list_mult_h)
#        res_mult = Mult_main(log_prob_t, log_prob_h, test_mat)
#        correct_rate = 1.0 - float(res_mult)/float(len(y_test))
#        print "alpha ", it, ", result ", res_mult, ", correct rate ", correct_rate

def reduce_volc_len(theta, dic_len):
    del_cols = []
    tmp = []
    for col in xrange(dic_len):
        if prob_list_mult_h[col] != 0 and prob_list_mult_t[col] != 0:
            dev = prob_list_mult_h[col] / prob_list_mult_t[col]
            tmp.append(dev)
            if dev >= 1/theta and dev <= theta:
                del_cols.append(col)
                
    dic_len = len(init_dic) - len(del_cols)   
         
    redwordscount_mat_h = np.delete(wordscount_mat_h, del_cols, axis=1)
    redwordscount_mat_t = np.delete(wordscount_mat_t, del_cols, axis=1)
    
    red_test_mat = np.delete(test_mat, del_cols, axis=1)
    
    red_sum_words_t = sum_count_words_bern(redwordscount_mat_t)
    red_sum_words_h = sum_count_words_bern(redwordscount_mat_h)
    red_sum_words_mult_t = sum_count_words_mult(redwordscount_mat_t)
    red_sum_words_mult_h = sum_count_words_mult(redwordscount_mat_h)
    
    # Bernoully model: Calculate each p(xi|y=trump) and (1-p(xi|y=trump)), then get log of them
    red_prob_list_bern_t = prob_list_Bern(red_sum_words_t, y_train, 1)
    red_one_prob_list_bern_t = np.add(1, np.negative(red_prob_list_bern_t))
    red_log_prob_list_bern_t = np.log(red_prob_list_bern_t)
    red_log_one_prob_list_bern_t = np.log(red_one_prob_list_bern_t)
    
    # Bernoully model: Calculate each p(xi|y=clinton) and (1-p(xi|y=clinton)), then get log of them
    red_prob_list_bern_h = prob_list_Bern(red_sum_words_h, y_train, 0)
    red_one_prob_list_bern_h = np.add(1, np.negative(red_prob_list_bern_h))
    red_log_prob_list_bern_h = np.log(red_prob_list_bern_h)
    red_log_one_prob_list_bern_h = np.log(red_one_prob_list_bern_h)

    # Multinomial model: Calculate each p(xi|y=trump), then get log of them
    red_prob_list_mult_t = prob_list_Mult(red_sum_words_mult_t, y_train, 1, 0.05)
    red_log_prob_list_mult_t = np.log(red_prob_list_mult_t)
    
    # Multinomial model: Calculate each p(xi|y=clinton), then get log of them
    red_prob_list_mult_h = prob_list_Mult(red_sum_words_mult_h, y_train, 0, 0.05)
    red_log_prob_list_mult_h = np.log(red_prob_list_mult_h)    
    
    # Get the results
    Bern_main(red_log_prob_list_bern_t, red_log_one_prob_list_bern_t, \
                         red_log_prob_list_bern_h, red_log_one_prob_list_bern_h, red_test_mat, "clintontrump.predictions.dev")
#    Bern_main(red_log_prob_list_bern_t, red_log_one_prob_list_bern_t, \
#                         red_log_prob_list_bern_h, red_log_one_prob_list_bern_h, red_test_mat, "clintontrump.predictions.test")
#    Mult_main(red_log_prob_list_mult_t, red_log_prob_list_mult_h, red_test_mat, "clintontrump.predictions.dev")
#    Mult_main(red_log_prob_list_mult_t, red_log_prob_list_mult_h, red_test_mat, "clintontrump.predictions.test")
    print "Vocabulary Size: ", dic_len
        
if __name__ == "__main__":
    # Data readin and dictionary length getting
    init_x_train = readin("clintontrump.bagofwords.train")
    init_x_test = readin("clintontrump.bagofwords.dev")
#    init_x_test = readin("clintontrump.bagofwords.test")
    init_y_train = readin("clintontrump.labels.train")
#    init_y_test = readin("clintontrump.labels.dev")
    init_dic = readin("clintontrump.vocabulary")
    
    dic_len = len(init_dic)

    # Data process
    x_train = convert_to_intlist(init_x_train)
    x_test = convert_to_intlist(init_x_test)
    
    y_train = trans_namelist(init_y_train)
#    y_test = trans_namelist(init_y_test)

    # Calculate words occurances for each twitter
    wordscount_mat_t = count_words(x_train, y_train, 1)
    wordscount_mat_h = count_words(x_train, y_train, 0)
    
    # Calculate p(y=trump) and p(y=clinton)
    prob_y_t = sum(y_train)/len(y_train)
    prob_y_h = 1 - prob_y_t
    
    # Get words occurances array
    sum_words_t = sum_count_words_bern(wordscount_mat_t)
    sum_words_h = sum_count_words_bern(wordscount_mat_h)
    sum_words_mult_t = sum_count_words_mult(wordscount_mat_t)
    sum_words_mult_h = sum_count_words_mult(wordscount_mat_h)
    
    # Bernoully model: Calculate each p(xi|y=trump) and (1-p(xi|y=trump)), then get log of them
    prob_list_bern_t = prob_list_Bern(sum_words_t, y_train, 1)
    one_prob_list_bern_t = np.add(1, np.negative(prob_list_bern_t))
    log_prob_list_bern_t = np.log(prob_list_bern_t)
    log_one_prob_list_bern_t = np.log(one_prob_list_bern_t)
    
    # Bernoully model: Calculate each p(xi|y=clinton) and (1-p(xi|y=clinton)), then get log of them
    prob_list_bern_h = prob_list_Bern(sum_words_h, y_train, 0)
    one_prob_list_bern_h = np.add(1, np.negative(prob_list_bern_h))
    log_prob_list_bern_h = np.log(prob_list_bern_h)
    log_one_prob_list_bern_h = np.log(one_prob_list_bern_h)

    # Multinomial model: Calculate each p(xi|y=trump) and (1-p(xi|y=trump)), then get log of them
    prob_list_mult_t = prob_list_Mult(sum_words_mult_t, y_train, 1, 0.05)
    log_prob_list_mult_t = np.log(prob_list_mult_t)
    
    # Multinomial model: Calculate each p(xi|y=clinton) and (1-p(xi|y=clinton)), then get log of them
    prob_list_mult_h = prob_list_Mult(sum_words_mult_h,y_train, 0, 0.05)
    log_prob_list_mult_h = np.log(prob_list_mult_h)

    # Idengyfy the top ten words
    top_ten_words()
    
    # Transfer x_test to multinomial-style matrix
    test_mat = count_words_test(x_test)    
    
    # Try vocabulary reduction.
#    reduce_volc_len(60, dic_len)
    
    Bern_main(log_prob_list_bern_t, log_one_prob_list_bern_t, log_prob_list_bern_h, log_one_prob_list_bern_h, test_mat, "clintontrump.predictions.dev")
#    Bern_main(log_prob_list_bern_t, log_one_prob_list_bern_t, log_prob_list_bern_h, log_one_prob_list_bern_h, test_mat, "clintontrump.predictions.test")
#    Mult_main(log_prob_list_mult_t, log_prob_list_mult_h, test_mat, "clintontrump.predictions.dev")
#    Mult_main(log_prob_list_mult_t, log_prob_list_mult_h, test_mat, "clintontrump.predictions.test")
    print "Prediction finished! \n Please check the clintontrump.predictions.dev file, thank you!"
    
    #For bonus credit
    N10 = sum_words_h
    N11 = sum_words_t
    N01 = np.add(sum(y_train), np.negative(sum_words_t))
    N00 = np.add((len(y_train)-sum(y_train)), np.negative(sum_words_h))
    N   = len(y_train)
    
    logPy = -(prob_y_t*np.log(prob_y_t)+ prob_y_h*np.log(prob_y_h))

    IG = []
    for i in xrange(dic_len):
        tmp = 0
        tmp += logPy
        if N10[i] == 0:
            term1 = 0
        else:
            term1 = (N10[i]/(N10[i]+N11[i]))*np.log(N10[i]/(N10[i]+N11[i]))
            
        if N11[i] == 0:
            term2 = 0
        else:
            term2 = (N11[i]/(N11[i]+N01[i]))*np.log(N11[i]/(N11[i]+N01[i]))
            
        if N00[i] == 0:
            term3 = 0
        else:
            term3 = (N00[i]/(N00[i]+N01[i]))*np.log(N00[i]/(N00[i]+N01[i]))
            
        if N01[i] == 0:
            term4 = 0
        else: term4 = (N01[i]/(N00[i]+N01[i]))*np.log(N01[i]/(N00[i]+N01[i]))
        
        tmp += (((N10[i]+N11[i])/N)*(term1 + term2) +  ((N00[i]+N01[i])/N)*(term3 + term4))
        IG.append(tmp)
    sort_IG_idx = sorted(range(len(IG)), key=IG.__getitem__)
    del_words = sort_IG_idx[0:3000]

  
#    MI  = []
#    for i in xrange(dic_len):
#        tmp = 0
#        
#        if N10[i] == 0:
#            term1 = 0
#        else:
#            term1 = (N10[i]/N)*np.log((N*N10[i])/((N11[i]+N10[i])*(N10[i]*N00[i])))
#            
#        if N11[i] == 0:
#            term2 = 0
#        else:
#            term2 = (N11[i]/N)*np.log((N*N11[i])/((N11[i]+N10[i])*(N11[i]*N01[i])))
#            
#        if N01[i] == 0:
#            term3 = 0
#        else:
#            term3 = (N01[i]/N)*np.log((N*N01[i])/((N01[i]+N00[i])*(N11[i]*N01[i])))
#            
#        if N00[i] == 0:
#            term4 = 0
#        else:
#            term4 = (N00[i]/N)*np.log((N*N00[i])/((N01[i]+N00[i])*(N10[i]*N00[i])))
#        
#        tmp += term1 + term2 + term3 + term4
#        MI.append(tmp)
#
#    sort_MI_idx = sorted(range(len(MI)), key=MI.__getitem__)
#    del_words = sort_MI_idx[:49:-1]

    #del_words.append(70)
    #del_words.append(4704)
    #del_words.append(4125)
    #del_words.append(4543)

    new_dic_len = len(init_dic) - len(del_words)
    
    new_wordscount_h = np.delete(wordscount_mat_h, del_words, axis=1)
    new_wordscount_t = np.delete(wordscount_mat_t, del_words, axis=1)
    new_test_mat = np.delete(test_mat, del_words, axis=1)
    
    new_sum_words_t = sum_count_words_bern(new_wordscount_t)
    new_sum_words_h = sum_count_words_bern(new_wordscount_h)
    new_sum_words_mult_t = sum_count_words_mult(new_wordscount_t)
    new_sum_words_mult_h = sum_count_words_mult(new_wordscount_h)
    
    new_prob_list_bern_t = prob_list_Bern(new_sum_words_t, y_train, 1)
    new_one_prob_list_bern_t = np.add(1, np.negative(new_prob_list_bern_t))
    new_log_prob_list_bern_t = np.log(new_prob_list_bern_t)
    new_log_one_prob_list_bern_t = np.log(new_one_prob_list_bern_t)
    
    new_prob_list_bern_h = prob_list_Bern(new_sum_words_h, y_train, 0)
    new_one_prob_list_bern_h = np.add(1, np.negative(new_prob_list_bern_h))
    new_log_prob_list_bern_h = np.log(new_prob_list_bern_h)
    new_log_one_prob_list_bern_h = np.log(new_one_prob_list_bern_h)
    
    new_prob_list_mult_t = prob_list_Mult(new_sum_words_mult_t, y_train, 1, 0.05)
    new_log_prob_list_mult_t = np.log(new_prob_list_mult_t)
    
    new_prob_list_mult_h = prob_list_Mult(new_sum_words_mult_h, y_train, 0, 0.05)
    new_log_prob_list_mult_h = np.log(new_prob_list_mult_h) 
    
#    Bern_main(new_log_prob_list_bern_t, new_log_one_prob_list_bern_t, \
#                         new_log_prob_list_bern_h, new_log_one_prob_list_bern_h, new_test_mat, "clintontrump.predictions.test")
#    Mult_main(new_log_prob_list_mult_t, new_log_prob_list_mult_h, new_test_mat, "clintontrump.predictions.test")
#    print "Prediction finished! \n Please check the clintontrump.predictions.test file, thank you!"
