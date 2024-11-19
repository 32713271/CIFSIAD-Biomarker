# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import pointbiserialr
from math import sqrt
import copy
import math
import warnings
# from pandas.errors import SettingWithCopyWarning
import time
pd.options.mode.chained_assignment = None
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

start_time = time.time()


def loadDataSet(fileName):
    print(f"Loading data from file: {fileName}")
    df = pd.read_csv(fileName)
    return df


def writesortedlist(filename, thelist):
    with open(filename, "w") as fw:
        for item in thelist:
            fw.write(item[0] + "\t" + str(item[1]) + "\n")


def writethelist(filename, thelist):
    with open(filename, "w") as fw:
        for item in thelist:
            fw.write(item + "\n")


def getdatadf(datafile):
    datadf = loadDataSet(datafile)
    labellist = datadf["Label"].tolist()
    del datadf["Label"]
    return datadf, labellist


def CFSmethod(datafile):
    datadf, labellist = getdatadf(datafile)
    print(datadf)
    selectdf = datadf.copy()
    allflist = datadf.columns.tolist()
    namelist = list(datadf.index)
    print(namelist)
    namelist = [int(var) for var in namelist]
    selectdf["class"] = namelist

    bestfset = calBFset(selectdf, allflist)
    writethelist(r"D:\jupyter\new_funtion\2024_7_9\7_30\duibi\o_bestfeatureda.txt", bestfset)


def calmulmerit(selectdf, sublist):
    retvalue = 0
    label = "class"
    k = len(sublist)
    namelist = list(selectdf["class"])
    classset = set(namelist)
    caldf = selectdf[sublist]
    allvalue = 0.0
    for feature in sublist:
        caldf = selectdf[sublist]
        middlevalue = 0.0
        for ind in classset:
            caldf[label] = np.where(selectdf[label] == ind, 1, 0)
            coeff = pointbiserialr(caldf[feature], caldf[label])
            middlevalue = abs(coeff.correlation) + middlevalue
        allvalue = middlevalue / float(len(classset)) + allvalue
    allvalue = allvalue / float(k)

    corr = selectdf[sublist].corr()
    corr.values[np.tril_indices_from(corr.values)] = np.nan
    corr = abs(corr)
    rff = corr.unstack().mean()
    retvalue = (k * allvalue) / sqrt(k + k * (k - 1) * rff)
    print(retvalue)
    return retvalue


def calBFset(selectdf, allflist):
    allfdict = getallfscoredict(selectdf, allflist)
    sortedflist = sorted(allfdict.items(), key=lambda item: item[1], reverse=True)
    writesortedlist(r"D:\jupyter\new_funtion\2024_7_9\7_30\duibi\o_CFS.txt", sortedflist)
    feaS = []
    feaS.append(sortedflist[0][0])
    maxvalue = sortedflist[0][1]
    for i in range(1, len(sortedflist)):
        print(str(i) + "/" + str(len(sortedflist)))
        itemf = sortedflist[i][0]
        feaS.append(itemf)
        newvalue = calmulmerit(selectdf, feaS)
        if newvalue > maxvalue:
            maxvalue = newvalue
        else:
            feaS.pop()
    print(feaS)
    return feaS


def getallfscoredict(selectdf, allflist):
    retdict = {}
    k = 1
    for f in allflist:
        print(k)
        k = k + 1
        score = calonemerit(selectdf, f)
        if math.isnan(score):
            continue
        retdict[f] = score
    return retdict


def calonemerit(selectdf, subname):
    retvalue = 0
    label = "class"
    namelist = list(selectdf["class"])
    classset = set(namelist)
    caldf = selectdf[subname].to_frame()
    allvalue = 0.0
    for ind in classset:
        caldf[label] = np.where(selectdf[label] == ind, 1, 0)
        coeff = pointbiserialr(caldf[subname], caldf[label])
        allvalue = abs(coeff.correlation) + allvalue
    allvalue = allvalue / float(len(classset))
    return allvalue


datafile = r"D:\jupyter\new_funtion\2024_7_9\7_30\mRNA_LncRNA_Non_Small_Cell_Lung_Carcinoma_data.csv"
CFSmethod(datafile)
end_time = time.time()

print("时间：", end_time - start_time)
