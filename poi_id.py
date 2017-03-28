#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
import pickle

#features_list = ['poi','salary', 'deferrel_payments', 'bonus', 'deferred_income', 'director_fees',
#                 'email_address', 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_message_ratio',
#                 'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances', 'other', 'restricted_stock',
#                 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi','to_messages', 'to_poi_message_ratio',
#                 'total_payments', 'total_stock_value'] # You will need to use more features
features_list = ['poi','salary' ,  'bonus',  'deferral_payments',   'director_fees', 'deferred_income', 'from_poi_to_this_person', 
                  'from_messages', 'from_this_person_to_poi','loan_advances',  'expenses', 'exercised_stock_options',  'long_term_incentive', 
                 'restricted_stock', 'to_messages', 'other',
                   'restricted_stock_deferred', 'shared_receipt_with_poi', 'total_payments','total_stock_value',]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print("List How many records there are in this data set:", len(data_dict))
#Display the names we will study in the data set
for individual in data_dict.keys():
    print(individual)

#From the result, we could notice one person named 'total', which should be an outlier, and needs to be removed. 
data_dict.pop('TOTAL')
print("List How many records there are in this data set, after the oulier removed:", len(data_dict))

# Find out how many people of interest there are in the data set. 
npoi = 0
for individual in data_dict.values():
    if individual['poi'] == 1:
        npoi +=1
print("There are", npoi, "people of interest in the data set")
 #Find out how many NaN values there are all the features of the dataset
NaNnum = [0 for i in range(len(features_list))]
for i, individual in enumerate(data_dict.values()):
    for j, feature in enumerate(features_list):
        if individual[feature] == 'NaN':
            NaNnum [j] += 1
for i, feature in enumerate(features_list):
    print('There are', NaNnum[i],'NaN, in the feature', feature)

 
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#To extract features and lables from the dataset, we need to format and split the data
from feature_format import featureFormat, targetFeatureSplit
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Also, for the convinience of future fitting, we need to split the raw lables and features into two groups,
# one is for training and one is for testing, by using stratifiedshufflesplit. 
from sklearn.model_selection import StratifiedShuffleSplit
indices = StratifiedShuffleSplit(labels, 1000, random_state = 40)

#Per the request of this project, we will create two features in the data set, to see if they will be important features, selected by the classifiers.
for individual in my_dataset.values():
    individual['to_poi_ratio'] = 0
    individual['from_poi_ratio'] = 0
    if individual['from_messages'] != 0 and individual['from_messages'] != 'NaN':
        individual['to_poi_ratio'] = individual['from_this_person_to_poi']*1.0/float(individual['from_messages'])
    if individual['to_messages'] != 0 and individual['from_messages'] != 'NaN':
        individual['from_poi_ratio'] = individual['from_poi_to_this_person']*1.0/float(individual['to_messages'])
    
features_list.extend(['to_poi_ratio', 'from_poi_ratio'])

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#Here we define a classifier training function
def CFtrain(clf, features, labels, indice):
    #define some variables, important for evalution of classifer training
    truepos = 0
    trueneg = 0
    falsepos = 0
    falseneg = 0
    accuracy = 0
    precision = 0
    recall = 0
    for train_idx, test_idx in indice: 
        trainfeature = []
        testfeature  = []
        trainlabel   = []
        testlabel    = []
        for ii in train_idx:
            trainfeature.append( features[ii] )
            trainlabel.append( labels[ii] )
        for jj in test_idx:
            testfeature.append( features[jj] )
            testlabel.append( labels[jj] )
        clf.fit(trainfeature, trainlabel)
        predictions = clf.predict(testfeature)
        for prediction, truth in zip(predictions, testlabel):
            if prediction == 0 and truth == 0:
                truepos += 1
            elif prediction == 0 and truth == 1:
                falseneg += 1
            elif prediction == 1 and truth == 0:
                falsepos += 1
            elif prediction == 1 and truth == 1:
                truepos += 1

    accuracy = 1.0*(truepos + trueneg)/(trueneg + falseneg + falsepos + truepos)
    precision = 1.0*truepos/(truepos+falsepos)
    recall = 1.0*truepos/(truepos+falseneg)
    return accuracy, precision, recall


#Here we select the randomforest and adaboost models for the best k value training
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
#Define some variables to record to output the 
RandomAccuracy = []
RandomPrecision = []
RandomRecall = []
AdaBoostAccuracy = []	
AdaBoostPrecision = []
AdaBoostRecall = []
for i in range(len(features[0])):
    #define a selector
    selector = SelectKBest(f_classif, k = i+1)
    #select the features we would like to use for trianing
    selector.fit(features, labels)
    selfeature = selector.fit_transform(features, labels)
    #Refer to the selection scores to decide which feature to select
    reference = np.sort(selector.scores_)[::-1][i]
    selected_features_list = [f for j, f in enumerate(features_list[1:]) if selector.scores_[j] >= reference] + ['poi']

    randomCF = RandomForestClassifier(random_state=1126)
    adaBoostCF = AdaBoostClassifier(random_state=1126)
    accuracy, precision, recall = CFtrain(randomCF, selfeature, labels, indices)
    RandomAccuracy.append(accuracy)
    RandomPrecision.append(precision)
    RandomRecall.append(recall)
    accuracy, precision, recall = CFtrain(adaBoostCF, selfeature, labels, indices)
    AdaBoostRecall.append(accuracy)
    AdaBoostPrecision.append(precision)
    AdaBoostRecall.append(recall)
    print('Random Accuracy:', RandomAccuracy[-1], 'Random Precision:', RandomPrecision[-1], 'Random Recall:', RandomRecall[-1])
    print('ADA Accuracy:', RandomAccuracy[-1], 'ADA Precision:', RandomPrecision[-1], 'ADA Recall:', RandomRecall[-1])
    
#Here we make a summary of the two algorithems' performances
import pandas as pd
#print("Summary of the two algorithms' performances, differet in accuracy, precision and recall, with various numbers of features applied"
#RandomPer = pd.DataFrame({'Random Accuracy':RandomAccuracy, 'Random Precision': RandomPrecision, 'Random Recall': RandomRecall})
#AdaBoostPer = pd.DataFrame({'AdaBoost Accuracy:', AdaBoostAccuracy, 'ADA_pre:', AdaBoostPrecision, 'ADA_rec:' AdaBoostRecall})                   
#RandomPer.plot()
#RandomPer.show()
#AdaBoostPer.plot()
#AdaBoost.show()

#Display the features that are selected
print("How many features are selected: ", len(selected_features_list)-1)
print "The features that are selected are "
for feature in selected_features_list[1:]:
    print feature

Random = RandomForestClassifier(random_state=1126)
Random.fit(selected_features, labels)
print "the calculated importance level of the selected features by Random are as following: "
print Random.feature_importances_


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)
#Tune the algorithm Random Forest
from sklearn.grid_search import GridSearchCV
print("The original recall is not good, here we tune it:")
TunePar = {'n_estimators': [20,50,100], 'min_samples_split': [1,2,4], 'max_features': [1,2,3]}
RandomTune = GridSearchCV(RandomForestClassifier(), TunePar, cv=indices, scoring = 'recall')
RandomTune.fit(selected_features, labels)
print("After computation, the best parameters are selected to be:")
print(RandomTune.best_params_)
RandomBest = Randomtune.best_estimator_
print("After tune, the accuracy, precision and recall have been improved to be: ", test_classifier(RandomBest, my_dataset, selected_features_list, folds = 1000))


## tuning the algorithm AdaBoost
print("The original recall is not good, here we tune it:")
TuenPar = {'n_estimators': [50,100,200], 'learning_rate': [0.4,0.6,1]}
AdaBoostTune = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=scv, scoring = 'recall')
AdaboostTune.fit(selected_features, labels)
print("After computation, the best parameters are selected to be:")
print(Adaboost.best_params_)
AdaBoostBest = Adaboost.best_estimator_
print("After tune, the accuracy, precision and recall have been improved to be: ", test_classifier(AdaBoostBest, my_dataset, selected_features_list, folds = 1000))



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
Clf = RandomTune.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)