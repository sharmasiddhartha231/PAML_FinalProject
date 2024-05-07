from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.datasets import make_classification
import seaborn as sns
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import plotly.graph_objects as go
import imblearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

df = pd.read_csv("Diabetes_Data_Sub_Strict_Main_String_New.txt", sep='\t')
df = df.drop(df[df.DIABETERES == 'Prediabetes'].index)
df.DIABETERES[df.DIABETERES == 'No Diabetes'] = 0
df.DIABETERES[df.DIABETERES == 'Diabetes'] = 1
cols = df.columns
cols = cols[0:32]
for i in cols:
  print(i)
  i = pd.get_dummies(df[i], drop_first=False)
  df = pd.concat([df,i], axis=1)

cols_new = df.columns
cols_new = cols_new[33:168]
df = pd.DataFrame(df, columns=cols_new)
df.columns = df.columns.astype(str)
#sex = pd.get_dummies(df['X_AGEG5YR'], drop_first=True)
#df.drop(['X_AGEG5YR'], axis=1, inplace=True)
#df = pd.concat([df,sex], axis=1)
#df = df.iloc[:,33:136]

x_data = df.drop(['DIABETERES'],axis=1)
y_data = df['DIABETERES']
y_data=y_data.astype('int')
x_data = x_data.replace(False,0, regex=True)
x_data = x_data.replace(True,1, regex=True)
#enc = OneHotEncoder(handle_unknown='ignore')
#enc.fit(x_data)

under = SMOTE(sampling_strategy=1)
steps = [('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
x_data, y_data = pipeline.fit_resample(x_data, y_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 0)
#X_train = enc.fit_transform(X_train).toarray()
#X_test = enc.fit_transform(X_test).toarray()

#gnb = GaussianNB()
#nb.fit(X_train, y_train)
#predictions_log = gnb.predict(X_test)
#print(classification_report(y_test, predictions_log))

##KNN
neigh = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
neigh.fit(X_train, y_train)
predictions_log = neigh.predict(X_test)
print(classification_report(y_test, predictions_log))

##Random Forest
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
predictions_log = clf.predict(X_test)
print(classification_report(y_test, predictions_log))

#Logistic Regression (from sklearn)
logmodel = LogisticRegression(penalty='l2', max_iter = 100000, solver = 'newton-cholesky')
logmodel.fit(X_train, y_train)
predictions_log = logmodel.predict(X_test)
print(classification_report(y_test, predictions_log))

#Decision tree
treed = DecisionTreeClassifier()
treed = treed.fit(X_train, y_train)
predictions_log = treed.predict(X_test)
print(classification_report(y_test, predictions_log))

#SVM
svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm.fit(X_train, y_train)
predictions_log = svm.predict(X_test)
print(classification_report(y_test, predictions_log))

#Linear SVM
lsvm = make_pipeline(StandardScaler(),LinearSVC(dual=False, random_state=0, tol=1e-5))
lsvm.fit(X_train, y_train)
predictions_log = lsvm.predict(X_test)
print(classification_report(y_test, predictions_log))

c1 = confusion_matrix(y_test, predictions_log, labels=[0,1])
tn, fp, fn, tp = c1.ravel()
specificity = tn / (tn+fp)
accuracy = (tp + tn)/(tp + tn + fp + fn)
misclassification = (fp + fn)/(tp + tn + fp + fn)
sensitivity = tp /(tp + fn)
precision = tp / (tp + fp)
accuracy
sensitivity
precision
specificity
