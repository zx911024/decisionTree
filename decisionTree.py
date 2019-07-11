#-*- coding:utf-8 -*-
from numpy import *
from scipy import *
import pandas as pd
from math import log
import operator
from sklearn import metrics
from platTree import createPlot

def calcShannonEnt(dataSet):
    '''
    计算信息熵
    '''
    numEntries = len(dataSet)
    # 类别字典（类别的名称为键，该类别的个数为值）
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # 还没添加到字典里的类型
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 求出每种类型的熵
    for key in labelCounts:
        # 每种类型个数占所有的比值
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt  #返回熵

def splitDataSet(dataSet, axis, value):
    '''
    按照给定的特征划分数据集
    '''
    retDataSet = []
    # 按dataSet矩阵中的第axis列的值等于value的分数据集
    for featVec in dataSet:
        # 值等于value的，每一行为新的列表（去除第axis个数据）
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  
            retDataSet.append(reducedFeatVec) 
    return retDataSet  #返回分类后的新矩阵

def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式
    '''
    # 求属性的个数
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    # 求所有属性的信息增益
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # 第i列属性的取值（不同值）数集合
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 求第i列属性每个不同值的熵*他们的概率
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i , value)
            # 求出该值在i列属性中的概率
            prob = len(subDataSet)/float(len(dataSet))
            # 求i列属性各值对于的熵求和
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 求出第i列属性的信息增益
        infoGain = baseEntropy - newEntropy
        # 保存信息增益最大的信息增益值以及所在的下表（列值i）
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain  
            bestFeature = i  

    return bestFeature

def majorityCnt(classList):
    '''
    找出出现次数最多的分类名称
    '''
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys(): classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    '''
    创建训练树
    '''
    # 创建需要创建树的训练数据的结果列表（例如最外层的列表是[N, N, Y, Y, Y, N, Y]）
    classList = [example[-1] for example in dataSet]

    # 如果所有的训练数据都是属于一个类别，则返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 训练数据只给出类别数据（没给任何属性值数据），返回出现次数最多的分类名称
    if (len(dataSet[0]) == 1):
        return majorityCnt(classList)

    # 选择信息增益最大的属性进行分（返回值是属性类型列表的下标）
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 根据下表找属性名称当树的根节点
    bestFeatLabel = labels[bestFeat]
    # 以bestFeatLabel为根节点建一个空树
    myTree = {bestFeatLabel:{}}
    # 从属性列表中删掉已经被选出来当根节点的属性
    del(labels[bestFeat])
    # 找出该属性所有训练数据的值（创建列表）
    featValues = [example[bestFeat] for example in dataSet]
    # 求出该属性的所有值得集合（集合的元素不能重复）
    uniqueVals = set(featValues)
    # 根据该属性的值求树的各个分支
    for value in uniqueVals:
        subLabels = labels[:]
        # 根据各个分支递归创建树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree  #生成的树

def classify(inputTree, featLabels, testVec):
    '''
    实用决策树进行分类
    '''
    firstStr = list(inputTree.keys())[0]  
    secondDict = inputTree[firstStr]  
    featIndex = featLabels.index(firstStr)  
    for key in secondDict.keys():  
        if testVec[featIndex] == key:  
            if type(secondDict[key]).__name__ == 'dict':  
                classLabel = classify(secondDict[key], featLabels, testVec)  
            else: classLabel = secondDict[key]  
    return classLabel

if __name__ == "__main__":
    # 读取训练集
    df=pd.read_csv('trainSet/gupiaonew.csv')
    # 取数据集
    data=df.values[:,1:].tolist()
    # 取数据列名称
    labels=df.columns.values[1:-1].tolist()
    # 取数据全列名
    labels_full=labels[:]
    # 决策树训练
    myTree = createTree(data,labels)
    # 决策树
    print ("--决策树--")
    print (myTree)
    # 测试集读取（GuPiaoTest.csv是测试集文件名称,其他数据测试，格式按照该文件输入）
    df=pd.read_csv('testSet/GupiaoTest.csv')
    testList = df.values[:,1:].tolist()
    # 开始测试，每条数据分别测试
    Y_test = []
    Y_pred = []
    for testData in testList:
        # 测试数据测试（输入数据为去掉类别的数据，即testData[:-1]）
        Y_test.append(testData[-1])
        dic = classify(myTree, labels_full, testData[:-1])
        Y_pred.append(dic)
        # print("-------------预测--------------")
        # print("输入数据："+str(testData[:-1]))
        # print("预测结果："+dic+' 实际结果：'+testData[-1])
    print("测试集类别：")
    print(Y_test)
    print("测试集预测结果：")
    print(Y_pred)
    # 将测试结果转换为DataFrame
    pre_y = pd.DataFrame(Y_pred)
    test_y = pd.DataFrame(Y_test)
    print("测试数据精确率、召回率、F1值：")
    print(metrics.classification_report(test_y, pre_y))
    # 绘制决策树
    createPlot(myTree)
