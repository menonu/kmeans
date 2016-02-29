#!/usr/bin/env python
# coding: utf-8

import sys,math,numpy
from sklearn.cluster import KMeans
from sklearn import metrics

from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE,SIG_DFL)

def measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==0:
           TN += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    if TP > FP:
        pass
    else :
        TP, FN = FN, TP
        FP, TN = TN, FP

    return(TP, FP, TN, FN)

if len(sys.argv) == 2:
    colfile = sys.argv[1]
else:
    colfile = 'collect.txt'

collecttxt = numpy.loadtxt(colfile, skiprows=1)
cfeatures = collecttxt[:,:-1]
clabels = collecttxt[:,-1].astype(numpy.int32)

kmeans_model = KMeans(n_clusters=2, random_state=100).fit(cfeatures)

labels = kmeans_model.labels_

perf = measure(clabels, labels)

print metrics.adjusted_rand_score(clabels, labels)
print 'Detection Rate' , float(perf[0])/(perf[0]+perf[3])
print 'False Positive Rate: ' ,float(perf[1])/(perf[1]+perf[2])

