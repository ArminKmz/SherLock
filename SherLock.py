__author__ = "Armin"

import csv
import nltk
import pickle
import re

learningSet = csv.reader(open("train.csv"))

# we have 5 class so, 5 classifier (opn, agr, ext, neu, con)
Features_con = []
Features_opn = []
Features_agr = []
Features_ext = []
Features_neu = []

users = []
allFeatures = []

pattern = re.compile(r"(.)\1{1,}", re.DOTALL)

def preProcess(status):
    re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '__LINK__', status)
    re.sub(r'\d+', '', status)
    return status
    
def FeaturesVector(status):
    words = set(status)
    features = {}
    for word in allFeatures:
        features[word] = (word in status)
    return features

def getFeatures(status):
    fv = []
    bagsOfWord = status.split()
    for word in bagsOfWord:
        word = word.strip('...?!')
        word = pattern.sub(r"\1\1\1", word)
        fv.append(word)
    return fv

def save_classifier(classifier, name):
    f = open(name+'.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
    
def load_classifier(name):
    f = open(name + '.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier

again = True
#mess = input("Do you want to load classifier? (y/n) ")
#if(mess == 'n'):
    #again = True
    
if(again):
    # extracting ans saving features
    for line in learningSet:
        userId = line[0]
        status = line[1]
        ext = line[7]
        neu = line[8]
        agr = line[9]
        con = line[10]
        opn = line[11]
        # date
        date = line[12]
        
        # network features
        networkSize = line[13]
        nBetweenness = line[15]
        density = line[16]
        nBrokerage = line[18]
        transitivity = line[19]
        if userId not in users:
            users.append(userId)
            # network size
            Features_con.append((networkSize, con))
            Features_ext.append((networkSize, ext))
            Features_agr.append((networkSize, agr))
            Features_neu.append((networkSize, neu))
            Features_opn.append((networkSize, opn))
            # normal betweenness
            Features_con.append((nBetweenness, con))
            Features_ext.append((nBetweenness, ext))
            Features_agr.append((nBetweenness, agr))
            Features_neu.append((nBetweenness, neu))
            Features_opn.append((nBetweenness, opn))
            # density
            Features_con.append((density, con))
            Features_ext.append((density, ext))
            Features_agr.append((density, agr))
            Features_neu.append((density, neu))
            Features_opn.append((density, opn))
            # normal brokerage
            Features_con.append((nBrokerage, con))
            Features_ext.append((nBrokerage, ext))
            Features_agr.append((nBrokerage, agr))
            Features_neu.append((nBrokerage, neu))
            Features_opn.append((nBrokerage, opn))
            # transitivity
            Features_con.append((transitivity, con))
            Features_ext.append((transitivity, ext))
            Features_agr.append((transitivity, agr))
            Features_neu.append((transitivity, neu))
            Features_opn.append((transitivity, opn))
            # add to all
            allFeatures.append(networkSize)
            allFeatures.append(nBetweenness)
            allFeatures.append(nBrokerage)
            allFeatures.append(transitivity)
            allFeatures.append(density)

        # date feature
        Features_con.append((date, con))
        Features_neu.append((date, neu))
        Features_agr.append((date, agr))
        Features_opn.append((date, opn))
        Features_ext.append((date, ext))
        allFeatures.append(date)
        
        # linguestic features
        status = preProcess(status)
        statusFeatures = getFeatures(status)
        allFeatures.extend(statusFeatures)
        Features_con.append((statusFeatures, con))
        Features_neu.append((statusFeatures, neu))
        Features_agr.append((statusFeatures, agr))
        Features_opn.append((statusFeatures, opn))
        Features_ext.append((statusFeatures, ext))
    allFeatures = list(set(allFeatures))
   
    print("Appling features...")
    # apply features
    con_trainingSet = nltk.classify.util.apply_features(FeaturesVector, Features_con)
    ext_trainingSet = nltk.classify.util.apply_features(FeaturesVector, Features_ext)
    neu_trainingSet = nltk.classify.util.apply_features(FeaturesVector, Features_neu)
    agr_trainingSet = nltk.classify.util.apply_features(FeaturesVector, Features_agr)
    opn_trainingSet = nltk.classify.util.apply_features(FeaturesVector, Features_opn)
    print("Features extracted.")
    print("Training classifiers...")
    # training classifiers
    NBC_con = nltk.NaiveBayesClassifier.train(con_trainingSet)
    print("Done 1 from 5.")
    NBC_ext = nltk.NaiveBayesClassifier.train(ext_trainingSet)
    print("Done 2 from 5.")
    NBC_neu = nltk.NaiveBayesClassifier.train(neu_trainingSet)
    print("Done 3 from 5.")
    NBC_agr = nltk.NaiveBayesClassifier.train(agr_trainingSet)
    print("Done 4 from 5.")
    NBC_opn = nltk.NaiveBayesClassifier.train(opn_trainingSet)
    print("Done 5 from 5.")
    print("Training classifiers done.")
    #print("Saving classifiers...")
    #save_classifier(NBC_con, "NBC_con")
    #save_classifier(NBC_ext, "NBC_ext")
    #save_classifier(NBC_neu, "NBC_neu")
    #save_classifier(NBC_agr, "NBC_agr")
    #save_classifier(NBC_opn, "NBC_opn")
else:
    print("loading classifiers ...")
    NBC_con = load_classifier("NBC_con")
    NBC_ext = load_classifier("NBC_ext")
    NBC_opn = load_classifier("NBC_opn")
    NBC_neu = load_classifier("NBC_neu")
    NBC_agr = load_classifier("NBC_agr")
    
 
Done = True

while(not Done):
    testIn = input("Status: ")
    nB = input("Normal Betweenness: ")
    nBr = input("Normal Brokerage: ")
    size = input("Normal Network Size: ")
    tr = input("Normal Transitivity: ")
    den = input("Normal Density: ")
    dt = input("Date :")
    FV = getFeatures(testIn);
    FV.append(nBr)
    FV.append(nB)
    FV.append(size)
    FV.append(tr)
    FV.append(den)
    FV.append(dt)
    classCon = NBC_con.classify(FeaturesVector(FV))
    classExt = NBC_ext.classify(FeaturesVector(FV))
    classAgr = NBC_agr.classify(FeaturesVector(FV))
    classNeu = NBC_neu.classify(FeaturesVector(FV))
    classOpn = NBC_opn.classify(FeaturesVector(FV))
    print("Extraversion : " + classExt)
    print("Neuroticism : " + classNeu)
    print("Agreeableness : " + classAgr)
    print("Conscientiousness : " + classCon)
    print("Openness : " + classOpn)
    mess = input("Do you want to countinue? (y/n) ")
    if mess == "n":
        Done = True

testingSet = csv.reader(open("test.csv"))

print("Evaluating ...")

tp_con = 0
tn_con = 0
fn_con = 0
fp_con = 0

tp_ext = 0
tn_ext = 0
fn_ext = 0
fp_ext = 0

tp_agr = 0
tn_agr = 0
fn_agr = 0
fp_agr = 0

tp_opn = 0
tn_opn = 0
fn_opn = 0
fp_opn = 0

tp_neu = 0
tn_neu = 0
fn_neu = 0
fp_neu = 0

for line in testingSet:
        testStatus = line[1]
        testExt = line[7]
        testNeu = line[8]
        testAgr = line[9]
        testCon = line[10]
        testOpn = line[11]
        testDate = line[12]
        testNetworkSize = line[13]
        testNBetweenness = line[15]
        testDensity = line[16]
        testNBrokerage = line[18]
        testTransitivity = line[19]
        FV = getFeatures(testStatus)
        FV.append(testNetworkSize)
        FV.append(testNBetweenness)
        FV.append(testDensity)
        FV.append(testNBrokerage)
        FV.append(testDate)
        FV.append(testTransitivity)
        classCon = NBC_con.classify(FeaturesVector(FV))
        classExt = NBC_ext.classify(FeaturesVector(FV))
        classAgr = NBC_agr.classify(FeaturesVector(FV))
        classNeu = NBC_neu.classify(FeaturesVector(FV))
        classOpn = NBC_opn.classify(FeaturesVector(FV))
        if classCon == testCon and testCon == 'y':
            tp_con += 1
        if classCon == testCon and testCon == 'n':
            tn_con += 1
        if classCon != testCon and testCon == 'y':
            fp_con += 1
        if classCon != testCon and testCon == 'n':
            fn_con += 1

        if classExt == testExt and testExt == 'y':
            tp_ext += 1
        if classExt == testExt and testExt == 'n':
            tn_ext += 1
        if classExt == testExt and testExt == 'y':
            fp_ext += 1
        if classExt == testExt and testExt == 'n':
            fn_ext += 1


        if classOpn == testOpn and testOpn == 'y':
            tp_opn += 1
        if classOpn == testOpn and testOpn == 'n':
            tn_opn += 1
        if classOpn == testOpn and testOpn == 'y':
            fp_opn += 1
        if classOpn == testOpn and testOpn == 'n':
            fn_opn += 1

        if classAgr == testAgr and testAgr == 'y':
            tp_agr += 1
        if classAgr == testAgr and testAgr == 'n':
            tn_agr += 1
        if classAgr == testAgr and testAgr == 'y':
            fp_agr += 1
        if classAgr == testAgr and testAgr == 'n':
            fn_agr += 1

        if classNeu == testNeu and testNeu == 'y':
            tp_neu += 1
        if classNeu == testNeu and testNeu == 'n':
            tn_neu += 1
        if classNeu == testNeu and testNeu == 'y':
            fp_neu += 1
        if classNeu == testNeu and testNeu == 'n':
            fn_neu += 1

Pre_opn = 0.5 * (tp_opn/(tp_opn+fp_opn) + tn_opn/(tn_opn+fn_opn))
Re_opn =  0.5 * (tp_opn/(tp_opn+fn_opn) + tn_opn/(tn_opn+fp_opn))
F1_opn = 2 * ((Pre_opn*Re_opn) / (Pre_opn+Re_opn))

Pre_agr = 0.5 * (tp_agr/(tp_agr+fp_agr) + tn_agr/(tn_agr+fn_agr))
Re_agr =  0.5 * (tp_agr/(tp_agr+fn_agr) + tn_agr/(tn_agr+fp_agr))
F1_agr = 2 * ((Pre_agr*Re_agr) / (Pre_agr+Re_agr))

Pre_ext = 0.5 * (tp_ext/(tp_ext+fp_ext) + tn_ext/(tn_ext+fn_ext))
Re_ext =  0.5 * (tp_ext/(tp_ext+fn_ext) + tn_ext/(tn_ext+fp_ext))
F1_ext = 2 * ((Pre_ext*Re_ext) / (Pre_ext+Re_ext))

Pre_neu = 0.5 * (tp_neu/(tp_neu+fp_neu) + tn_neu/(tn_neu+fn_neu))
Re_neu =  0.5 * (tp_neu/(tp_neu+fn_neu) + tn_neu/(tn_neu+fp_neu))
F1_neu = 2 * ((Pre_neu*Re_neu) / (Pre_neu+Re_neu))

Pre_con = 0.5 * (tp_con/(tp_con+fp_con) + tn_con/(tn_con+fn_con))
Re_con =  0.5 * (tp_con/(tp_con+fn_con) + tn_con/(tn_con+fp_con))
F1_con = 2 * ((Pre_con*Re_con) / (Pre_con+Re_con))


print "Con Pre(avg) = " + str(Pre_con)
print "Con Re(avg) = " + str(Re_con)
print "Con F1(avg) = " + str(F1_con)

print "Agr Pre(avg) = " + str(Pre_agr)
print "Agr Re(avg) = " + str(Re_agr)
print "Agr F1(avg) = " + str(F1_agr)

print "Ext Pre(avg) = " + str(Pre_ext)
print "Ext Re(avg) = " + str(Re_ext)
print "Ext F1(avg) = " + str(F1_ext)

print "Neu Pre(avg) = " + str(Pre_neu)
print "Neu Re(avg) = " + str(Re_neu)
print "Neu F1(avg) = " + str(F1_neu)

print "Opn Pre(avg) = " + str(Pre_opn)
print "Opn Re(avg) = " + str(Re_opn)
print "Opn F1(avg) = " + str(F1_opn)

