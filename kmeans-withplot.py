#!/usr/bin/env python
# coding: utf-8

import sys,math,struct,os,numpy
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE,SIG_DFL)


if len(sys.argv) == 2:
    colfile = sys.argv[1]
else:
    colfile = 'collect.txt'
collecttxt = numpy.loadtxt(colfile, skiprows=1)
cfeatures = collecttxt[:,:-1]
clabels = collecttxt[:,-1].astype(numpy.int32)

cwheel = pd.read_csv(colfile, delim_whitespace=True)
#target = cwheel.ix[:,-1]
target = cwheel.ix[:,len(cwheel.ix[0])-1]
print target
#features = numpy.loadtxt('vector.txt')

kmeans_model = KMeans(n_clusters=2**3, random_state=100).fit(cfeatures)

labels = kmeans_model.labels_

colortable = {
        0:'#FE2EF7',
        1:'#A9F5F2',
        2:'#FF0000',
        3:'#00FF00',
        4:'#2E2EFE',
        5:'#DF7401',
        6:'#FFFF00',
        7:'#424242'
        }

colors = target.map(lambda x: colortable.get(x))
#print colors

print
print 'similarity:' , metrics.adjusted_rand_score(clabels, labels)

#scatter_matrix(data, color = colors, alpha = 0.6, diagonal = 'hist')
if len(sys.argv) == 4:
    axisx = int(sys.argv[1])
    axisy = int(sys.argv[2])
    axisz = int(sys.argv[3])
elif len(sys.argv) == 5:
    axisx = int(sys.argv[2])
    axisy = int(sys.argv[3])
    axisz = int(sys.argv[4])
else:
    axisx = 0
    axisy = 1
    axisz = 2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cfeatures[:,axisx],cfeatures[:,axisy],cfeatures[:,axisz],c = colors)
ax.set_xlabel('RX'+str(axisx))
ax.set_ylabel('RX'+str(axisy))
ax.set_zlabel('RX'+str(axisz))

plt.show()

