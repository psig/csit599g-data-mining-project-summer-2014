""""
  mining.py: Decision Tree Classifier

"""
import time
import csv
import math

#for testing script run time
start_time = time.perf_counter()

#open CSV containing data
with open('clean-data-subset.csv') as file:
  reader = csv.reader(file)
  #load data into list of tuples
  columns = ('Age','Gender','Income','Education','Tobacco',
             'Alcohol','IllegalDrugs','Depression',
             'AlcoholOrDrugTreatment','MentalHealthTreatment',
             'ArrestedOrJailed')
  data = [tuple(row) for row in reader]

#split data into training and testing lists


  #check for base cases

  #for each attribute, find the info gain for splitting on that attribute

  #create a decision node that splits on the attribute with highest gain

  #apply steps to each sublist

def entropy(data):
  """Returns entropy of dataset based on class label in final position """
  totalRecords = len(data)
  classRecords = sum(1 for element in data if element[len(element)-1] == '1')
  nonClassRecords = totalRecords - classRecords
  entropy = ( - (classRecords/totalRecords) 
              * math.log(classRecords/totalRecords)
              - (nonClassRecords/totalRecords) 
              * math.log(nonClassRecords/totalRecords))
  return(entropy)

def information_gain(data, element):
  """Returns information gain for splitting data on specified element """
  currentEntropy = entropy(data)

  #split data into sublists
  sublists = {}
  for row in data:
    if row[element] not in sublists:
      sublists[row[element]] = [row]
    else:
      sublists[row[element]].append(row)
  
  #get entropy for each sublist
  sublistEntropy = 0
  for key in sublists:
    sublistEntropy += entropy(sublists[key]) * (len(sublists[key]) / len(data))
  return(currentEntropy - sublistEntropy)

print("Initial entropy:",entropy(data))
for x in range(0, 10):
  print("Info gain for splitting on ",columns[x],": ",information_gain(data, x))
print(time.perf_counter() - start_time)