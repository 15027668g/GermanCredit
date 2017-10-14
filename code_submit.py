# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn import tree

from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest

#import data
dataset = pd.read_csv("german.data", sep = "\s+",index_col = False, header=None)

#Name header for each column
dataset.columns = ["Status of existing checking account",
                   "Duration in month",
                   "Credit history",
                   "Purpose",
                   "Credit amount",
                   "Savings account/bonds",
                   "Present employment since",
                   "Installment rate in percentage of disposable income",
                   "Personal status and sex",
                   "Other debtors / guarantors",
                   "Present residence since",
                   "Property",
                   "Age in years",
                   "Other installment plans",
                   "Housing",
                   "Number of existing credits at this bank",
                   "Job",
                   "Number of people being liable to provide maintenance for",
                   "Telephone",
                   "foreign worker",
                   "Customer quality result"
                   ]
#output a csv with header (for reference only)
dataset.to_csv("csvfile")

#Map qualitative data to quantitative data
#Seperate label and feature
data_norm = dataset[["Status of existing checking account", 
                   "Duration in month",
                   "Credit history",
                   "Purpose",
                   "Credit amount",
                   "Savings account/bonds",
                   "Present employment since",
                   "Installment rate in percentage of disposable income",
                   "Personal status and sex",
                   "Other debtors / guarantors",
                   "Present residence since",
                   "Property",
                   "Age in years",
                   "Other installment plans",
                   "Housing",
                   "Number of existing credits at this bank",
                   "Job",
                   "Number of people being liable to provide maintenance for",
                   "Telephone",
                   "foreign worker",
                   ]].copy()
data_label = dataset[["Customer quality result"]].copy()
data_norm['Status of existing checking account'] = data_norm['Status of existing checking account'].map({'A11': 1, 'A12': 2, 'A13': 3,'A14': 4})
data_norm['Credit history'] = data_norm['Credit history'].map({'A30': 0, 'A31': 1, 'A32': 2,'A33': 3,'A34': 4})
data_norm['Purpose'] = data_norm['Purpose'].map({'A40': 0, 'A41': 1, 'A42': 2,'A43': 3,'A44': 4,'A45': 5,'A46': 6,'A47': 7,'A48': 8,'A49': 9,'A410': 10})
data_norm['Savings account/bonds'] = data_norm['Savings account/bonds'].map({'A61': 1, 'A62': 2, 'A63': 3,'A64': 4,'A65': 5})
data_norm['Present employment since'] = data_norm['Present employment since'].map({'A71': 1, 'A72': 2, 'A73': 3,'A74': 4,'A75': 5})
data_norm['Personal status and sex'] = data_norm['Personal status and sex'].map({'A91': 1, 'A92': 2, 'A93': 3,'A94': 4,'A95': 5})
data_norm['Other debtors / guarantors'] = data_norm['Other debtors / guarantors'].map({'A101': 1, 'A102': 2, 'A103': 3})
data_norm['Property'] = data_norm['Property'].map({'A121': 1, 'A122': 2, 'A123': 3, 'A124': 4})
data_norm['Other installment plans'] = data_norm['Other installment plans'].map({'A141': 1, 'A142': 2, 'A143': 3})
data_norm['Housing'] = data_norm['Housing'].map({'A151': 1, 'A152': 2, 'A153': 3})
data_norm['Job'] = data_norm['Job'].map({'A171': 1, 'A172': 2, 'A173': 3,  'A174': 4})
data_norm['Telephone'] = data_norm['Telephone'].map({'A191': 1, 'A192': 2})
data_norm['foreign worker'] = data_norm['foreign worker'].map({'A201': 1, 'A202': 2})

#Calculate Basic statistic value 
data_norm_mean = data_norm.mean() #mean
data_norm_std = data_norm.std() #standard deviation
data_norm_summary = pd.DataFrame(data_norm_mean, columns = ['avg'])
data_norm_summary.insert(loc=1,column="std",value=data_norm_std)

#VarianceThreshold feature selection (for reference,to see different to Chi2)
def VarianceThreshold_selector(data):
    columns = data.columns
    selector = VarianceThreshold(.5)
    selector.fit_transform(data)
    labels = [columns[x] for x in selector.get_support(indices=True)]
    return pd.DataFrame(selector.fit_transform(data), columns=labels)
data_sel1 = VarianceThreshold_selector(data_norm)

#Min-Max Normalization for numerical data
data_norm['Duration in month'] = data_norm['Duration in month'].sub(data_norm['Duration in month'].min()).div((data_norm['Duration in month'].max() - data_norm['Duration in month'].min()))
data_norm['Credit amount'] = data_norm['Credit amount'].sub(data_norm['Credit amount'].min()).div((data_norm['Credit amount'].max() - data_norm['Credit amount'].min()))
data_norm['Age in years'] = data_norm['Age in years'].sub(data_norm['Age in years'].min()).div((data_norm['Age in years'].max() - data_norm['Age in years'].min()))

#Chi-Square feature selection
def SelectKBest_selector(data,label):
    columns = data.columns
    selector = SelectKBest(chi2, k=12)
    selector.fit(data,label)
    labels = [columns[x] for x in selector.get_support(indices=True)]
    return pd.DataFrame(selector.fit_transform(data,label), columns=labels), selector.scores_

#Calculate Basic statistic value (Chi2)
data_sel2, score= SelectKBest_selector(data_norm, data_label)
data_norm_summary.insert(loc=2,column="chi2",value=score)
data_norm_summary = data_norm_summary.round(3)
data_norm_summary.to_csv("data summary.csv")


# Split dataset to training data and testing data (900 for train, 100 for test)
data_train, data_test, label_train, label_test = train_test_split(data_sel2, data_label, test_size=100, train_size=900, random_state=42)

#KNN algorithm
model = KNeighborsClassifier(n_neighbors=32)
model.fit(data_train, label_train.values.ravel())
predictions = model.predict(data_test)
# KNN prediction Score
print("KNN Prediction Score: ",metrics.accuracy_score(label_test, predictions))


#Decision Tree
model = tree.DecisionTreeClassifier(random_state=50,max_depth=None, criterion="entropy")
clf = model.fit(data_train,label_train)
dotfile = open("dtree.dot", 'w') #generate dot file for the tree
tree.export_graphviz(clf, out_file = dotfile, feature_names = data_train.columns)
dotfile.close()

#Decision Tree prediction Score
tree_predict = model.predict(data_test)
print("Decision Tree Prediction Score : ", metrics.accuracy_score(label_test, tree_predict))
"""
Open the .dot file created with a text editor. 
Copy all the text in the .dot file and paste to the textbox at http://webgraphviz.com/ (Web Service of Graphviz)
"""
'''
Association rule of Aprori result is generated from Orange3 visualization software
''' 
data_aprori = data_sel2.copy()
#data_aprori['Duration in month'] = '<40'
data_aprori['Duration in month'][data_aprori['Duration in month'] >= 0.8] = 4
data_aprori['Duration in month'][(data_aprori['Duration in month'] >= 0.6) & (data_aprori['Duration in month'] < 0.8)] = 3
data_aprori['Duration in month'][(data_aprori['Duration in month'] >= 0.4) & (data_aprori['Duration in month'] < 0.6)] = 2
data_aprori['Duration in month'][(data_aprori['Duration in month'] >= 0.2) & (data_aprori['Duration in month'] < 0.4)] = 1
data_aprori['Duration in month'][data_aprori['Duration in month'] < 0.2] = 0

data_aprori['Credit amount'][data_aprori['Credit amount'] >= 0.8] = 4
data_aprori['Credit amount'][(data_aprori['Credit amount'] >= 0.6) & (data_aprori['Credit amount'] < 0.8)] = 3
data_aprori['Credit amount'][(data_aprori['Credit amount'] >= 0.4) & (data_aprori['Credit amount'] < 0.6)] = 2
data_aprori['Credit amount'][(data_aprori['Credit amount'] >= 0.2) & (data_aprori['Credit amount'] < 0.4)] = 1
data_aprori['Credit amount'][data_aprori['Credit amount'] < 0.2] = 0

data_aprori['Age in years'][data_aprori['Age in years'] >= 0.8] = 4
data_aprori['Age in years'][(data_aprori['Age in years'] >= 0.6) & (data_aprori['Age in years'] < 0.8)] = 3
data_aprori['Age in years'][(data_aprori['Age in years'] >= 0.4) & (data_aprori['Age in years'] < 0.6)] = 2
data_aprori['Age in years'][(data_aprori['Age in years'] >= 0.2) & (data_aprori['Age in years'] < 0.4)] = 1
data_aprori['Age in years'][data_aprori['Age in years'] < 0.2] = 0

#generate data for aprori algo in Orange 3
data_aprori.insert(loc=12,column="Customer quality result",value=data_label)

data_aprori.to_csv("partition.csv")

def Scaling(maxValue,minValue,targetvalue):
    m = (maxValue - minValue)/1
    return (m*targetvalue+minValue)

#Partitioning
print("Duration in month Max: ",(dataset["Duration in month"].max()))
print("Duration in month Min: ",(dataset["Duration in month"].min()))
print(Scaling(dataset["Duration in month"].max(),dataset["Duration in month"].min(),0))
print(Scaling(dataset["Duration in month"].max(),dataset["Duration in month"].min(),0.2))
print(Scaling(dataset["Duration in month"].max(),dataset["Duration in month"].min(),0.4))
print(Scaling(dataset["Duration in month"].max(),dataset["Duration in month"].min(),0.6))
print(Scaling(dataset["Duration in month"].max(),dataset["Duration in month"].min(),0.8))
print(Scaling(dataset["Duration in month"].max(),dataset["Duration in month"].min(),1.0))

print("Credit amount Max: ",(dataset["Credit amount"].max()))
print("Credit amount Min: ",(dataset["Credit amount"].min()))
print(Scaling(dataset["Credit amount"].max(),dataset["Credit amount"].min(),0))
print(Scaling(dataset["Credit amount"].max(),dataset["Credit amount"].min(),0.2))
print(Scaling(dataset["Credit amount"].max(),dataset["Credit amount"].min(),0.4))
print(Scaling(dataset["Credit amount"].max(),dataset["Credit amount"].min(),0.6))
print(Scaling(dataset["Credit amount"].max(),dataset["Credit amount"].min(),0.8))
print(Scaling(dataset["Credit amount"].max(),dataset["Credit amount"].min(),1.0))

print("Age in years Max: ",(dataset["Age in years"].max()))
print("Age in years Min: ",(dataset["Age in years"].min()))
print(Scaling(dataset["Age in years"].max(),dataset["Age in years"].min(),0))
print(Scaling(dataset["Age in years"].max(),dataset["Age in years"].min(),0.2))
print(Scaling(dataset["Age in years"].max(),dataset["Age in years"].min(),0.4))
print(Scaling(dataset["Age in years"].max(),dataset["Age in years"].min(),0.6))
print(Scaling(dataset["Age in years"].max(),dataset["Age in years"].min(),0.8))
print(Scaling(dataset["Age in years"].max(),dataset["Age in years"].min(),1.0))