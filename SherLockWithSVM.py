__author__ = "Armin"

import csv
import svm
from svmutil import *
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

def makeDict(features, featureList):
    sortedFeatures = sorted(featureList)
    feature_vector = []
    labels = []
    for t in features:
        label = 0
        Map = {}
        for w in sortedFeatures:
            Map[w] = 0

        words = t[0]
        c = t[1]
        for word in words:
            if word in Map:
                Map[word] = 1
                
        values = Map.values()
        feature_vector.append(values)
        if(c == 'y'):
            label = 1
        elif(c == 'n'):
            label = 0
        labels.append(label)
    return {'feature_vector' : feature_vector, 'labels': labels}

def makeDictTest(features, featuresList):
    sortedFeatures = sorted(featuresList)
    feature_vector = []
    for t in features:
        Map = {}
        for w in sortedFeatures:
            Map[w] = 0

        words = t
        for word in words:
            if word in Map:
                Map[word] = 1
                
        values = Map.values()
        feature_vector.append(values)

    return feature_vector
    

def svm(features, allFeatures):
    res = makeDict(features, allFeatures)
    problem = svm_problem(res['labels'], res['feature_vector'])
    param = svm_parameter('-q')
    param.kernel_type = LINEAR
    classifier = svm_train(problem, param)
    return classifier
    
def yesOrNo(resList):
    counter = 0;
    for i in resList:
        if i == 0.0:
            counter += 1
    if(counter > len(resList)/2):
        return 'n'
    return 'y'

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
   

    print("Training classifiers...")
    # training classifiers
    SVM_CON = svm(Features_con, allFeatures)
    print("Done 1 from 5.")
    SVM_AGR = svm(Features_agr, allFeatures)
    print("Done 2 from 5.")
    SVM_NEU = svm(Features_neu, allFeatures)
    print("Done 3 from 5.")
    SVM_EXT = svm(Features_ext, allFeatures)
    print("Done 4 from 5.")
    SVM_OPN = svm(Features_opn, allFeatures)
    print("Done 5 from 5.")
    print("Training classifiers done.")

else:
    print("loading classifiers ...")

    
 
Done = False

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
    fv = makeDictTest(FV, allFeatures)
    con_label, con_acc, con_val = svm_predict([0] * len(fv),fv, SVM_CON)
    ext_label, ext_acc, ext_val =  svm_predict([0] * len(fv),fv, SVM_EXT)
    agr_label, agr_acc, agr_val = svm_predict([0] * len(fv),fv, SVM_AGR)
    neu_label, neu_acc, neu_val = svm_predict([0] * len(fv),fv, SVM_NEU)
    opn_label, opn_acc, opn_val = svm_predict([0] * len(fv),fv, SVM_OPN)
    print(con_label)
    print("Extraversion : " + str(yesOrNo(ext_label)))
    print("Neuroticism : " + str(yesOrNo(neu_label)))
    print("Agreeableness : " + str(yesOrNo(agr_label)))
    print("Conscientiousness : " + str(yesOrNo(con_label)))
    print("Openness : " + str(yesOrNo(opn_label)))
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

def precision(tp, tn, fp, fn):
    return 0.5 * (tp/(tp+fp) + tn/(tn+fn))

def recall(tp, tn, fp, fn):
    return 0.5 * (tp/(tp+fn) + tn/(tn+fp))

def F1(tp, tn, fp, fn):
    pre = precision(tp, tn, fp, fn)
    re = recall(tp, tn, fp, fn)
    return 2 * (pre * re / (pre + re))


print "Con Pre(avg) = " + str(precision(tp_con, tn_con, fp_con, fn_con))
print "Con Re(avg) = " + str(recall(tp_con, tn_con, fp_con, fn_con))
print "Con F1(avg) = " + str(F1(tp_con, tn_con, fp_con, fn_con))

print "Agr Pre(avg) = " + str(precision(tp_agr, tn_agr, fp_agr, fn_agr))
print "Agr Re(avg) = " + str(recall(tp_agr, tn_agr, fp_agr, fn_agr))
print "Agr F1(avg) = " + str(F1(tp_agr, tn_agr, fp_agr, fn_agr))

print "Ext Pre(avg) = " + str(precision(tp_ext, tn_ext, fp_ext, fn_ext))
print "Ext Re(avg) = " + str(recall(tp_ext, tn_ext, fp_ext, fn_ext))
print "Ext F1(avg) = " + str(F1(tp_ext, tn_ext, fp_ext, fn_ext))

print "Neu Pre(avg) = " + str(precision(tp_neu, tn_neu, fp_neu, fn_neu))
print "Neu Re(avg) = " + str(recall(tp_neu, tn_neu, fp_neu, fn_neu))
print "Neu F1(avg) = " + str(F1(tp_neu, tn_neu, fp_neu, fn_neu))

print "Opn Pre(avg) = " + str(precision(tp_opn, tn_opn, fp_opn, fn_opn))
print "Opn Re(avg) = " + str(recall(tp_opn, tn_opn, fp_opn, fn_opn))
print "Opn F1(avg) = " + str(F1(tp_opn, tn_opn, fp_opn, fn_opn))

