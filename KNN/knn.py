"""
Created on  2018-02-26
@author: Jingchao Yang

A basic k-NN classification and the condensed 1-NN algorithm for the Letter Recognition
Some credit goes to https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
"""
import csv
import random
import math
import operator
import pandas as pd


# function for data loading
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        # print(dataset)
        for x in range(len(dataset) - 1):
            # convert all data to float
            for y in range(16):
                dataset[x][y] = float(dataset[x][y])
            # split the data to 10 groups, 9 are using for training
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# using euclidean distance for searching knn
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# filter to select neighbors
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# majority of the neighbor attribute goes to the testing instance
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    # sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def getconsistentSet(trainingSet, k):
    """
    To reduce the number of training samples
    Retain only the samples that are needed to define the decision boundary
    A subset of the training data that correctly classifies all of the original training data

    :param trainingSet:
    :return:
    """
    consistentSet = []
    consistentSet.append(trainingSet[0])  # Initialize subset with a single training example
    trainingSet = trainingSet[1:]
    predictions = []
    # Classify all samples using the subset, and transfer an incorrectly classified sample to the subset
    for i in range(len(trainingSet)):
        neighbors = getNeighbors(consistentSet, trainingSet[i], k)
        result = getResponse(neighbors)
        predictions.append(result)
        if repr(result) != repr(trainingSet[i][-1]):
            consistentSet.append(trainingSet[i])
            print('adding %s to consistentSet' % neighbors[0])
    return consistentSet


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.75
    loadDataset('letter-recognition.data.csv', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 1
    # consistentSet = []  # a subset of the training data that correctly classifies all of the original training data
    consistentSet = getconsistentSet(trainingSet, k)
    consistentSet.to_csv('consistentSet.csv')
    for x in range(len(testSet)):
        neighbors = getNeighbors(consistentSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

        # if repr(result) == repr(testSet[x][-1]):
        #     consistentSet.append(neighbors[0])

    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

    # consistentSet_ = pd.DataFrame(consistentSet)
    # consistentSet_.to_csv('consistentSet.csv')
    # print('condense set saved for letter training')


main()
