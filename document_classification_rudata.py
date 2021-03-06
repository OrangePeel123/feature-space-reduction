
from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt


###DBSCAN libs
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

###Classification and 20news libs
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

import gensim
from gensim.models import KeyedVectors
#from gensim.test.utils import datapath
import pymorphy2
import re
import os
import random




# parse commandline arguments
op = OptionParser()
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--mi_select",
              action="store", type="int", dest="select_mi",
              help="Select some number of features using a mutual information test")
op.add_option("--lsa",
              action="store", type="int", dest="select_lsa",
              help="Select some number of features using a lsa")
op.add_option("--embedding",
              action="store", type="int", dest="select_emb",
              help="text embedding")
op.add_option("--cluster",
              action="store", type="int", dest="select_cluster",
              help="Select some number of features using a clustering")



def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()



print("Loading dataset ")


train_data = []
test_data = []
train_target = []
test_target = []
startdir_name = "rudata/results"
category = 0
cantread = 0
for root, dirs, files in os.walk(startdir_name):
    for d in dirs:
        if d!='norm':
            corpus = []
            target = []
            listoffiles = os.listdir(startdir_name+"/"+d+"/norm")
            for f in listoffiles:
                #print("dir %s  file %s" % (d, f))
                try:
                    file_name = startdir_name+ "/"+d +"/norm"+"/" + f
                    my_file = open(file_name, encoding="utf-8"
                        )
                    my_file_contents = my_file.read()
                    my_file.close()
                    r = my_file_contents.replace("{"," ")
                    r = r.replace("}"," ")
                    corpus.append(r)
                    target.append(category)
                except:
                    cantread=cantread+1
                #print(r)
            category = category + 1
            train_data = train_data + corpus[len(corpus)//3:]
            test_data = test_data + corpus[:len(corpus)//3]
            train_target = train_target + target[len(target)//3:]
            test_target = test_target + target[:len(target)//3]

   
shuffle_train = list(zip(train_data,train_target))
random.shuffle(shuffle_train)
train_data,train_target = zip(*shuffle_train)

shuffle_test = list(zip(test_data,test_target))
random.shuffle(shuffle_test)
test_data,test_target = zip(*shuffle_test)



print('data loaded')
#print(data_train)

# split a training set and a test set
y_train, y_test = train_target, test_target
print(y_train,len(y_train))
#print(re.split("[,.() \-!?:]+",data_train.data[0]))
print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(sublinear_tf=True,
                             max_df=0.5,
                             min_df=2
                             )
#vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_base_train = X_train
print(X_train.shape)
##########shape - kol-vo dokov, kol-vo priznakov

print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(test_data)
X_base_test = X_test
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

if opts.select_emb:
    ###########making train vectors with w2v or ft#############
    wrong = 0
    all_words = 0
    morph = pymorphy2.MorphAnalyzer()
    #model = gensim.models.KeyedVectors.load_word2vec_format("rumodel300/1/model.bin", binary=True)
    model = KeyedVectors.load_word2vec_format("rumodel300/cc.ru.300.vec")
    ALL_train = []
    for text in train_data:
        words = 0
        t = text.replace("\n"," ")
        t = re.split("[,.() \-!?:]+",t)
        txt = np.full(300,0)
        for w in t:
            words = words + 1
            try:
                txt = txt + model.wv[w]
            except:
                words = words - 1
                wrong = wrong + 1
        ALL_train.append(txt/words)
        all_words = all_words + words
    #print(len(ALL_train))
    #print(wrong)
    #print(all_words)
    ###################vectors done##########
    #############making test vectors#############
    ALL_test = []
    for text in test_data:
        words = 0
        t = text.replace("\n"," ")
        t = re.split("[,.() \-!?:]+",t)
        txt = np.full(300,0)
        for w in t:
            words = words + 1
            try:
                txt = txt + model.wv[w]
            except:
                words = words - 1
                wrong = wrong + 1
        ALL_test.append(txt/words)
        all_words = all_words + words
    X_train = ALL_train
    X_test = ALL_test
    #print(len(ALL_test))
    #print(wrong)
    #print(all_words)
###########################




##############CLUSTER features###########

if opts.select_cluster:
##################category features########
    vectorizer = CountVectorizer()
    X_train_c = vectorizer.fit_transform(train_data)
    #X_test_c = vectorizer.transform(data_test.data)
    X = X_train_c.toarray()
    vec0 = np.full(X_train_c.shape[1],0)
    vec1 = np.full(X_train_c.shape[1],0)
    vec2 = np.full(X_train_c.shape[1],0)
    vec3 = np.full(X_train_c.shape[1],0)

    vec4 = np.full(X_train_c.shape[1],0)
    vec5 = np.full(X_train_c.shape[1],0)
    vec6 = np.full(X_train_c.shape[1],0)
    vec7 = np.full(X_train_c.shape[1],0)

    vec8 = np.full(X_train_c.shape[1],0)
    vec9 = np.full(X_train_c.shape[1],0)
    vec10 = np.full(X_train_c.shape[1],0)
    vec11 = np.full(X_train_c.shape[1],0)

    vec12 = np.full(X_train_c.shape[1],0)
    vec13 = np.full(X_train_c.shape[1],0)
    vec14 = np.full(X_train_c.shape[1],0)
    vec15 = np.full(X_train_c.shape[1],0)

    vec16 = np.full(X_train_c.shape[1],0)
    vec17 = np.full(X_train_c.shape[1],0)
    vec18 = np.full(X_train_c.shape[1],0)
    vec19 = np.full(X_train_c.shape[1],0)

    vec20 = np.full(X_train_c.shape[1],0)
    vec21 = np.full(X_train_c.shape[1],0)
    vec22 = np.full(X_train_c.shape[1],0)
    vec23 = np.full(X_train_c.shape[1],0)

    vec24 = np.full(X_train_c.shape[1],0)
    vec25 = np.full(X_train_c.shape[1],0)
    vec26 = np.full(X_train_c.shape[1],0)
    vec27 = np.full(X_train_c.shape[1],0)
    vec28 = np.full(X_train_c.shape[1],0)

    for i in range(len(y_train)):
        if y_train[i] == 0:
          vec0 = vec0 + X[i]
        elif y_train[i] == 1:
          vec1 = vec1 + X[i]
        elif y_train[i] == 2:
          vec2 = vec2 + X[i]
        elif y_train[i] == 3:
          vec3 = vec3 + X[i]
        elif y_train[i] == 4:
          vec4 = vec4 + X[i]
        elif y_train[i] == 5:
          vec5 = vec5 + X[i]
        elif y_train[i] == 6:
          vec6 = vec6 + X[i]
        elif y_train[i] == 7:
          vec7 = vec7 + X[i]
        elif y_train[i] == 8:
          vec8 = vec8 + X[i]
        elif y_train[i] == 9:
          vec9 = vec9 + X[i]
        elif y_train[i] == 10:
          vec10 = vec10 + X[i]
        elif y_train[i] == 11:
          vec11 = vec11 + X[i]
        elif y_train[i] == 12:
          vec12 = vec12 + X[i]
        elif y_train[i] == 13:
          vec13 = vec13 + X[i]
        elif y_train[i] == 14:
          vec14 = vec14 + X[i]
        elif y_train[i] == 15:
          vec15 = vec15 + X[i]
        elif y_train[i] == 16:
          vec16 = vec16 + X[i]
        elif y_train[i] == 17:
          vec17 = vec17 + X[i]
        elif y_train[i] == 18:
          vec18 = vec18 + X[i]
        elif y_train[i] == 19:
          vec19 = vec19 + X[i]
        elif y_train[i] == 20:
          vec20 = vec20 + X[i]
        elif y_train[i] == 21:
          vec21 = vec21 + X[i]
        elif y_train[i] == 22:
          vec22 = vec22 + X[i]
        elif y_train[i] == 23:
          vec23 = vec23 + X[i]
        elif y_train[i] == 24:
          vec24 = vec24 + X[i]
        elif y_train[i] == 25:
          vec25 = vec25 + X[i]
        elif y_train[i] == 26:
          vec26 = vec26 + X[i]
        elif y_train[i] == 27:
          vec27 = vec27 + X[i]
        elif y_train[i] == 28:
          vec28 = vec28 + X[i]
    category_feat = []
    for i in range(X_train_c.shape[1]):
        category_feat.append([vec0[i], vec1[i], vec2[i], vec3[i], vec4[i],
         vec5[i], vec6[i], vec6[i], vec7[i], vec8[i], vec9[i],
          vec10[i], vec11[i], vec12[i], vec13[i],vec14[i] ,
          vec15[i] ,vec16[i] ,vec17[i] ,vec18[i] ,vec19[i] ,
          vec20[i] ,vec21[i] ,vec22[i] ,vec23[i] ,vec24[i] ,
          vec25[i] , vec26[i], vec27[i], vec28[i]])
    #########features W2V############
    #model = gensim.models.KeyedVectors.load_word2vec_format("engmodel300/GoogleNews-vectors-negative300.bin", binary=True)
#    model = KeyedVectors.load_word2vec_format("engmodel300/wiki-news-300d-1M.vec")
#    features = []
#    x=0
#    for w in vectorizer.get_feature_names():
#    #    x=x+1
#        try:
#            f = model.wv[w]
#
#        except:
#            f = np.full(300,0)
#            #print(w)
#        features.append(f)
#    features = np.concatenate((features,category_feat), axis = 1)
#    #print(features[0], len(features[0]))
#    #exit()
    kmeans = KMeans(n_clusters = opts.select_cluster, random_state=0).fit(category_feat)
    labels = kmeans.labels_
    #clustering = DBSCAN(eps=0.7, min_samples=2).fit(features)
    #labels = clustering.labels_
    print("cluster done")
    ##############combine features###########
    cluster_train = []
    Xarr = X#_train_c.toarray()
    for doc in Xarr:
      count = 0
      cluster_doc = np.full(opts.select_cluster,0)
      for t in doc:
        i = labels[count]
        cluster_doc[i] = cluster_doc[i] + t
        count = count + 1
      cluster_train.append(cluster_doc)
    X_train = cluster_train
    print("CLUSTER n_samples: %d, n_features: %d" % (len(cluster_train), len(cluster_train[0])))
    
    cluster_test = []
    Xarr = vectorizer.transform(test_data).toarray()
    for doc in Xarr:
      count = 0
      cluster_doc = np.full(opts.select_cluster,0)
      for t in doc:
        i = labels[count]
        cluster_doc[i] = cluster_doc[i] + t
        count = count + 1
      cluster_test.append(cluster_doc)
    X_test = cluster_test
    print("CLUSTER n_samples: %d, n_features: %d" % (len(cluster_test), len(cluster_test[0])))

  
#c = 0
#for i in labels:
#  if i!=0:
#    c = c + 1
#print(c)
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#print("clusters: %d" % n_clusters_)
#exit()

#################################
# mapping from integer feature name to original token string

feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)
####lsa
print("lsa")
t0 = time()
if opts.select_lsa:
    svd = TruncatedSVD(opts.select_lsa)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X_train_lsa = lsa.fit_transform(X_train)
    X_test_lsa = lsa.transform(X_test)
    X_train = X_train_lsa
    X_test = X_test_lsa 
    duration = time() - t0
    print("lsa in %f" % duration)


 
if opts.select_mi:
    print("Extracting %d best features by a mutual info test" %
          opts.select_mi)
    t0 = time()
    minfo = SelectKBest(mutual_info_classif, k=opts.select_mi)
    X_train = minfo.fit_transform(X_train, y_train)
    X_test = minfo.transform(X_test)
    X_mi_train = X_train
    X_mi_test = X_test
    # keep selected feature names
    feature_names = [feature_names[i] for i
                     in minfo.get_support(indices=True)]
    print(feature_names[:100])
    print("done in %fs" % (time() - t0))
    print()


if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    chi2_train = X_train
    chi2_test = X_test
    # keep selected feature names
    feature_names = [feature_names[i] for i
                     in ch2.get_support(indices=True)]
    print(feature_names[:100])
    
    print("done in %fs" % (time() - t0))
    print()

target_names = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    X_train = X_base_train
    X_test = X_base_test
    if opts.select_emb:
        X_train = ALL_train
        X_test = ALL_test
#        X_train = emb_train
#        X_test = emb_test
    if opts.select_cluster:
        X_train = cluster_train
        X_test = cluster_test
    if opts.select_lsa:
        X_train = X_train_lsa
        X_test = X_test_lsa  
    if opts.select_mi:
        X_train = X_mi_train
        X_test = X_mi_test
    if opts.select_chi2:
        X_train = chi2_train
        X_test = chi2_test

    clf.fit(X_train, y_train) #X_train, ALL_train, cluster_train
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)  #X_test, ALL_test, cluster_test
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print()

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
clf, name = (KNeighborsClassifier(n_neighbors=10), "kNN")
print('=' * 80)
print(name)
results.append(benchmark(clf))

print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

print('=' * 80)
print("Naive Bayes")
results.append(benchmark(BernoulliNB(alpha=.01)))
