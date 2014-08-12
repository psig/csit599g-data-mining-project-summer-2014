""""
  mining.py: Decision Tree Classifier
  
  Based on ID3 algorithm as described on:
  http://www.onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html
  and
  http://nbviewer.ipython.org/github/gumption/Python_for_Data_Science/blob/master/1_Introduction.ipynb
  
"""
import time
import csv
import math
import random
import collections


#for testing script run time
start_time = time.perf_counter()


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
    remainingAttributes = attributes.remove(nextBestAttribute)

    for attribute in partitions:
      #build subtree for each partition
      subtree = build_decision_tree(partitions[attribute],
                                    remainingAttributes,
                                    defaultClass)
      
      #add subtree to parent
      tree[nextBestAttribute][attribute] = subtree

  return tree


def entropy(data):
  """Returns entropy of dataset based on class label in final position"""
  totalRecords = len(data)
  classRecords = sum(1 for element in data if element[-1] == '1')
  nonClassRecords = totalRecords - classRecords
  entropy = ( - (classRecords/totalRecords) 
              * math.log(classRecords/totalRecords)
              - (nonClassRecords/totalRecords) 
              * math.log(nonClassRecords/totalRecords))
  return(entropy)

  
def split_data(data, attribute):
  """Returns a dictionary of data subsets after splitting on attribute"""
  sublists = {}
  for row in data:
    if row[attribute] not in sublists:
      sublists[row[attribute]] = [row]
    else:
      sublists[row[attribute]].append(row)
  return(sublists)


def information_gain(data, attribute):
  """Returns information gain for splitting data on specified attribute"""
  currentEntropy = entropy(data)

  #split data into sublists
  sublists = split_data(data, attribute)
  
  #get entropy for each sublist
  sublistEntropy = 0
  for key in sublists:
    sublistEntropy += entropy(sublists[key]) * (len(sublists[key])/len(data))
  return(currentEntropy - sublistEntropy)


def choose_best_attribute(data, attributes):
  """Returns the attribute in a list of attributes that would offer the
     greatest information gain if it is used to split the data list"""
  bestAttribute = -1
  informationGain = 0
  for attribute in attributes:
    currentInformationGain = information_gain(data, attribute)
    if currentInformationGain > informationGain:
      bestAttribute = attribute
      informationGain = currentInformationGain
  return(bestAttribute)

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


#test code
"""
print("Initial entropy:",entropy(data))
for x in range(0, 10):
  print("Info gain for splitting on ",columns[x],": ",information_gain(data,x))
print(choose_best_attribute(data, range(0,10)))
"""
print(build_decision_tree(trainingData, list(range(0,10))))

#display script run time
print(time.perf_counter() - start_time)