from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import plotly.graph_objects as go
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

df = pd.read_csv("Diabetes_Data_Sub_Strict_Main_String_New.txt", sep='\t')
df = df.drop(df[df.DIABETERES == 'Prediabetes'].index)
df.DIABETERES[df.DIABETERES == 'No Diabetes'] = 0
df.DIABETERES[df.DIABETERES == 'Diabetes'] = 1
df = df.reset_index(drop=True)
#df = df[['DIABETERES','X_AGEG5YR','X_RACE']]
#cols = ['X_AGEG5YR','X_RACE']
cols = df.columns
cols = cols[0:33]
for i in cols:
  print(i)
  i = pd.get_dummies(df[i], drop_first=False)
  df = pd.concat([df,i], axis=1)

cols_new = df.columns
cols_new = cols_new[33:173]
df = pd.DataFrame(df, columns=cols_new)
df.columns = df.columns.astype(str)
#sex = pd.get_dummies(df['X_AGEG5YR'], drop_first=True)
#df.drop(['X_AGEG5YR'], axis=1, inplace=True)
#df = pd.concat([df,sex], axis=1)
#df = df.iloc[:,33:136]

x_data = df.drop(['DIABETERES'],axis=1)
#x_data = x_data.drop(['X_AGEG5YR'],axis=1)
#x_data = x_data.drop(['X_RACE'],axis=1)
y_data = df['DIABETERES']
y_data=y_data.astype('int')
x_data = x_data.replace(False,0, regex=True)
x_data = x_data.replace(True,1, regex=True)
#enc = OneHotEncoder(handle_unknown='ignore')
#enc.fit(x_data)

over = SMOTE(sampling_strategy=0.25)
under = RandomUnderSampler(sampling_strategy=0.5)
over1 = SMOTE(sampling_strategy=1)
under1 = SMOTE(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 0)
x_data1, y_data1 = pipeline.fit_resample(x_data, y_data)
X_train1, X_test1, y_train1, y_test1 = train_test_split(x_data1, y_data1, test_size = 0.3, random_state = 0)
x_data2, y_data2 = over1.fit_resample(x_data, y_data)
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_data2, y_data2, test_size = 0.3, random_state = 0)
x_data3, y_data3 = under1.fit_resample(x_data, y_data)
X_train3, X_test3, y_train3, y_test3 = train_test_split(x_data3, y_data3, test_size = 0.3, random_state = 0)

#X_train = enc.fit_transform(X_train).toarray()
#X_test = enc.fit_transform(X_test).toarray()

gnb = GaussianNB()
gnb.fit(X_train3, y_train3)
y_pred = gnb.predict(X_test)
c2 = confusion_matrix(y_test, y_pred, labels=[0,1])
tn, fp, fn, tp = c2.ravel()
specificity = tn / (tn+fp)
accuracy = (tp + tn)/(tp + tn + fp + fn)
misclassification = (fp + fn)/(tp + tn + fp + fn)
sensitivity = tp /(tp + fn)
precision = tp / (tp + fp)
accuracy
sensitivity
precision
specificity
#print(classification_report(y_test, y_pred))

##KNN
neigh = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
neigh.fit(X_train3, y_train3)
y_pred = neigh.predict(X_test)
c2 = confusion_matrix(y_test, y_pred, labels=[0,1])
tn, fp, fn, tp = c2.ravel()
specificity = tn / (tn+fp)
accuracy = (tp + tn)/(tp + tn + fp + fn)
misclassification = (fp + fn)/(tp + tn + fp + fn)
sensitivity = tp /(tp + fn)
precision = tp / (tp + fp)
accuracy
sensitivity
precision
specificity
#print(classification_report(y_test, y_pred))

##Random Forest
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train3, y_train3)
y_pred = clf.predict(X_test)
c2 = confusion_matrix(y_test, y_pred, labels=[0,1])
tn, fp, fn, tp = c2.ravel()
specificity = tn / (tn+fp)
accuracy = (tp + tn)/(tp + tn + fp + fn)
misclassification = (fp + fn)/(tp + tn + fp + fn)
sensitivity = tp /(tp + fn)
precision = tp / (tp + fp)
accuracy
sensitivity
precision
specificity
#print(classification_report(y_test, y_pred))

#Logistic Regression (from sklearn)
#logmodel = SGDClassifier(max_iter = 1000, alpha=0.001)
logmodel = LogisticRegression(max_iter = 10000, solver = 'saga', C=0.1, tol=0.001)
logmodel.fit(X_train1, y_train1)
y_pred = logmodel.predict(X_test)
c2 = confusion_matrix(y_test, y_pred, labels=[0,1])
tn, fp, fn, tp = c2.ravel()
specificity = tn / (tn+fp)
accuracy = (tp + tn)/(tp + tn + fp + fn)
misclassification = (fp + fn)/(tp + tn + fp + fn)
sensitivity = tp /(tp + fn)
precision = tp / (tp + fp)
accuracy
sensitivity
precision
specificity
#y_pred1 = logmodel.predict(X_train)
#print(classification_report(y_test, y_pred))

#Decision tree
treed = DecisionTreeClassifier()
treed = treed.fit(X_train3, y_train3)
y_pred = treed.predict(X_test)
c2 = confusion_matrix(y_test, y_pred, labels=[0,1])
tn, fp, fn, tp = c2.ravel()
specificity = tn / (tn+fp)
accuracy = (tp + tn)/(tp + tn + fp + fn)
misclassification = (fp + fn)/(tp + tn + fp + fn)
sensitivity = tp /(tp + fn)
precision = tp / (tp + fp)
accuracy
sensitivity
precision
specificity
#SVM
svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm.fit(X_train1, y_train1)
y_pred = svm.predict(X_test)
#print(classification_report(y_test, y_pred))

#Linear SVM
lsvm = make_pipeline(StandardScaler(),LinearSVC(dual=False, random_state=0, tol=1e-5))
lsvm.fit(X_train3, y_train3)
y_pred = lsvm.predict(X_test)
#print(classification_report(y_test, y_pred))

c1 = confusion_matrix(y_test, y_pred, labels=[0,1])
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

c2 = confusion_matrix(y_train, y_pred1, labels=[0,1])
tn, fp, fn, tp = c2.ravel()
specificity = tn / (tn+fp)
accuracy = (tp + tn)/(tp + tn + fp + fn)
misclassification = (fp + fn)/(tp + tn + fp + fn)
sensitivity = tp /(tp + fn)
precision = tp / (tp + fp)
accuracy
sensitivity
precision
specificity