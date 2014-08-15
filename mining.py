""""
  mining.py: Decision Tree Classifier  
  Based on ID3 algorithm as described on:
  http://www.onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html
  and
  http://nbviewer.ipython.org/github/gumption/Python_for_Data_Science/blob/master/1_Introduction.ipynb
  
  Peter Sigur
  ps18@hood.edu
  CSIT 599G / Dr. Liu / Summer II 2014
  
  August 15, 2014
"""
import time
import csv
import math
import random
import collections

#for testing script run time
start_time = time.perf_counter()


def classify_testing_set(tree, data):
  """Returns success rate of classifying data with supplied tree"""
  successful = 0
  unsuccessful = 0
  for record in data:
    predictedLabel = classify(tree, record)
    actualLabel = record[-1]
    if predictedLabel:
      if predictedLabel[0][0] == actualLabel:
        successful += 1
      else:
        unsuccessful += 1

  return successful / (successful + unsuccessful)  


def classify(tree, data, defaultClass=None):
  """Returns a classification label for each record in data, 
     based on the supplied tree"""
  if not tree:
    return defaultClass
    
  if not isinstance(tree, dict):
    return tree
    
  attributeIndex = list(tree.keys())[0]
  attributeValues = list(tree.values())[0]
  incomingDataValue = data[attributeIndex]
  if incomingDataValue not in attributeValues:
    return defaultClass
    
  return classify(attributeValues[incomingDataValue], data, defaultClass)


def build_decision_tree(data, attributes, defaultClass=None):
  """Recursively builds a decision tree based on the ID3 algorithm"""
  classCounts = collections.Counter([record[-1] for record in data])

  #empty dataset or empty attributes list
  if not data or not attributes:
    return defaultClass
  
  #if entire dataset has same class label, return that class label
  elif len(classCounts) == 1:
    return data[0][-1]

  else:
    defaultClass = classCounts.most_common(1)
    
    #select next attribute to split on
    nextBestAttribute = choose_best_attribute(data, attributes)
    
    #create tree with initial deciding attribute
    tree = {nextBestAttribute:{}}
    
    #partition dataset based on attribute split
    partitions = split_data(data, nextBestAttribute)
    
    #remove attribute from set to split on
    attributes.remove(nextBestAttribute)

    for attribute in partitions:
      #build subtree for each partition
      subtree = build_decision_tree(partitions[attribute],
                                    attributes,
                                    defaultClass)
      
      #add subtree to parent
      tree[nextBestAttribute][attribute] = subtree

  return tree


def entropy(data):
  """Returns entropy of dataset based on class label in final position"""
  totalRecords = len(data)
  classRecords = sum(1 for element in data if element[-1] == '1')
  nonClassRecords = totalRecords - classRecords
  if classRecords == 0 or nonClassRecords == 0:
    return 0
  else:
    entropy = ( - (classRecords/totalRecords) 
              * math.log(classRecords/totalRecords, 2)
              - (nonClassRecords/totalRecords) 
              * math.log(nonClassRecords/totalRecords, 2))
    return entropy

  
def split_data(data, attribute):
  """Returns a dictionary of data subsets after splitting on attribute"""
  sublists = {}
  for row in data:
    if row[attribute] == '-1':
      continue
    elif row[attribute] not in sublists:
      sublists[row[attribute]] = [row]
    else:
      sublists[row[attribute]].append(row)
  return sublists


def information_gain(data, attribute):
  """Returns information gain for splitting data on specified attribute"""
  currentEntropy = entropy(data)

  #split data into sublists
  sublists = split_data(data, attribute)
  
  #get entropy for each sublist
  sublistEntropy = 0
  for key in sublists:
    sublistEntropy += entropy(sublists[key]) * (len(sublists[key])/len(data))
  return currentEntropy - sublistEntropy


def choose_best_attribute(data, attributes):
  """Returns the attribute in a list of attributes that would offer the
     greatest information gain if it is used to split the data list"""
  bestAttribute = attributes[0]
  informationGain = 0
  for attribute in attributes:
    currentInformationGain = information_gain(data, attribute)
    if currentInformationGain > informationGain:
      bestAttribute = attribute
      informationGain = currentInformationGain
  return bestAttribute


#open CSV containing data
with open('clean-data-subset.csv') as file:
  reader = csv.reader(file)
  #load data into list of tuples
  columns = ('Age','Gender','Income','Education','Tobacco',
             'Alcohol','IllegalDrugs','Depression',
             'AlcoholOrDrugTreatment','MentalHealthTreatment',
             'ArrestedOrJailed')
  data = [tuple(row) for row in reader]

#split data into training and testing lists (75% training, 25% testing)
random.shuffle(data)
breakPoint = int((len(data)*.75))
trainingData = data[:breakPoint]
testingData = data[breakPoint:]

#build classifying decision tree based on training set
decisionTree = build_decision_tree(trainingData, list(range(0,10)))

#run test set through decision tree
successRate = classify_testing_set(decisionTree, testingData)

print("Success rate was:",successRate * 100)
      
#display script run time
print("Run time:",time.perf_counter() - start_time)