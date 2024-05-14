import numpy as np                    
from sklearn.model_selection import train_test_split
import streamlit as st                  
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import pickle
import os
import seaborn as sns
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
#############################################

st.markdown('### Predicting your Diabetes Status')

st.markdown('Now that we have seen the overall results and trained various models and gotten an idea of how they work and what works best, we created this interface to allow the users to predict whether they have diabetes or not. For this, the users will enter their information below, choose a model to run and it will predict whether the user has Diabetes or not. This is specifically based on providing all information as we have not included any functionality for predicting whether the user has diabetes or not based on a subset of factors.')

# Helper Function
def load_dataset(filepath):
    """
    This function uses the filepath (string) a .csv file locally on a computer 
    to import a dataset with pandas read_csv() function. Then, store the 
    dataset in session_state.

    Input: data is the filename or path to file (string)
    Output: pandas dataframe df
    """
    data = pd.read_csv(filepath, sep='\t')
    st.session_state['data'] = data
    return data

def split_dataset_predict(df, number, input_row,sample_opt=1,oversample_val=0.25, undersample_val=0.5,random_state=42):
    # Add code here
    df = df.drop(df[df.DIABETERES == 'Prediabetes'].index)
    df.DIABETERES[df.DIABETERES == 'No Diabetes'] = 0
    df.DIABETERES[df.DIABETERES == 'Diabetes'] = 1
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    X, y = df.loc[:, ~df.columns.isin(['DIABETERES'])], df.loc[:, df.columns.isin(['DIABETERES'])]
    X = pd.concat([X, pd.DataFrame([input_row])], ignore_index=True) 
    col_vals = X.columns
    for i in col_vals:
        i = pd.get_dummies(X[i], drop_first=False)
        X = pd.concat([X,i], axis=1)
    X = X.loc[:, ~X.columns.isin(col_vals)]
    X.columns = X.columns.astype(str)
    X = X.replace(False,0, regex=True)
    X = X.replace(True,1, regex=True)
    y=y.astype('int')
    X_predict = X.tail(1)
    #X.drop(X.tail(1).index,inplace=True)
    #over = SMOTE(sampling_strategy=oversample_val)
    #under = RandomUnderSampler(sampling_strategy=undersample_val)
    #steps = [('o', over), ('u', under)]
    #pipeline = Pipeline(steps=steps)
    #X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(X, y, test_size=number/100, random_state=random_state, stratify=y)
    #if sample_opt == 1:
    #    X_train, X_test, y_train, y_test = train_test_split(X_train_main, y_train_main, test_size=number/100, random_state=random_state, stratify=y_train_main)    
    #if sample_opt == 2:
    #    X,y = over.fit_resample(X, y)
    #    X_train, X_test, y_train, y_test = train_test_split(X_train_main, y_train_main, test_size=number/100, random_state=random_state, stratify=y_train_main)
    #if sample_opt == 3:
    #    X,y = under.fit_resample(X, y)
    #    X_train, X_test, y_train, y_test = train_test_split(X_train_main, y_train_main, test_size=number/100, random_state=random_state, stratify=y_train_main)
    #if sample_opt == 4:
    #X,y = pipeline.fit_resample(X, y)
    #X_train, X_test, y_train, y_test = train_test_split(X_train_main, y_train_main, test_size=number/100, random_state=random_state, stratify=y_train_main)
    
    return X_predict

class LogisticRegression_GD(object):
    def __init__(self, learning_rate=0.001, num_iterations=1000): 
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.likelihood_history=[]
    def predict_probability(self, X):
        score = np.dot(X, self.W) + self.b
        y_pred = 1. / (1.+np.exp(-score)) 
        return y_pred
    def compute_avg_log_likelihood(self, X, Y, W):
        #indicator = (Y==+1)
        #scores = np.dot(X, W) 
        #logexp = np.log(1. + np.exp(-scores))
        #mask = np.isinf(logexp)
        #logexp[mask] = -scores[mask]
        #lp = np.sum((indicator-1)*scores - logexp)/len(X)
        scores = np.dot(X, W) 
        score = 1 / (1 + np.exp(-scores))
        y1 = ((Y * np.log(score)))
        y2 = ((1-Y) * np.log(1 - score))
        lp = -np.mean(y1 + y2)
        return lp
    def update_weights(self):      
        num_examples, num_features = self.X.shape
        y_pred = self.predict(self.X)
        dW = self.X.T.dot(self.Y-y_pred) / num_examples 
        db = np.sum(self.Y-y_pred) / num_examples 
        self.b = self.b + self.learning_rate * db
        self.W = self.W + self.learning_rate * dW
        log_likelihood=0
        log_likelihood += self.compute_avg_log_likelihood(self.X, self.Y, self.W)
        self.likelihood_history.append(log_likelihood)
    def predict(self, X):
        y_pred= 0
        scores = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        y_pred = [0 if z <= 0.5 else +1 for z in scores]
        return y_pred 
    def fit(self, X, Y):   
        self.X = X
        self.Y = Y
        num_examples, num_features = self.X.shape    
        self.W = np.zeros(num_features)
        self.b = 0
        self.likelihood_history=[]
        for _ in range(self.num_iterations):          
            self.update_weights()  
    def get_weights(self):
            out_dict = {'Logistic Regression': []}
            W = np.array([f for f in self.W])
            out_dict['Logistic Regression'] = self.W
            return out_dict

class LogisticRegression_SGD(LogisticRegression_GD):
    def __init__(self, num_iterations, learning_rate, batch_size): 
        self.likelihood_history=[]
        self.batch_size=batch_size
        # invoking the __init__ of the parent class
        LogisticRegression_GD.__init__(self, learning_rate, num_iterations)
    def fit(self, X, Y):
        permutation = np.random.permutation(len(X))
        self.X = X[permutation,:]
        self.Y = Y[permutation]
        self.num_features, self.num_examples = self.X.shape    
        W = np.zeros(self.num_examples)
        self.W = W
        b = 0
        self.b = b
        likelihood_history = []
        i = 0 
        self.likelihood_history = likelihood_history 
        for itr in range(self.num_iterations):
            predictions = self.predict_probability(self.X[i:i+self.batch_size,:])
            indicator = (self.Y[i:i+self.batch_size]==+1)
            errors = indicator - predictions
            for j in range(len(self.W)):
                dW = errors.dot(self.X[i:i+self.batch_size,j].T)
                self.W[j] += self.learning_rate * dW 
            lp = self.compute_avg_log_likelihood(self.X[i:i+self.batch_size,:], Y[i:i+self.batch_size],
                                        self.W)
            self.likelihood_history.append(lp)
            i += self.batch_size
            if i+self.batch_size > len(self.X):
                permutation = np.random.permutation(len(self.X))
                self.X = self.X[permutation,:]
                self.Y = self.Y[permutation]
                i = 0
            self.learning_rate=self.learning_rate/1.02
###################### FETCH DATASET #######################
df = None
if('data' in st.session_state):
    df = st.session_state['data']
else:
    current_working_directory = os.getcwd()
    filepath=os.path.join(current_working_directory, 'Diabetes_Data_Sub_Strict_Main_String_New.txt')
    if(filepath):
        df = load_dataset(filepath)

######################### MAIN BODY #########################

######################### EXPLORE DATASET #########################

if df is not None:
    # Restore dataset if already in memory
    st.session_state['data'] = df

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 1. Enter your Information')
    lg_col1, lg_col2 = st.columns(2)
    with (lg_col1):
        SEXVAR = st.selectbox(
            label='Sex of respondent',
            options=['Male', 'Female'],
            key='SEXVAR'
        )
        X_AGEG5YR = st.selectbox(
            label='Age Group:',
            options=['18-24 YO','25-29 YO','30-34 YO','35-39 YO','40-44 YO','45-49 YO','50-54 YO','55-59 YO','60-64 YO','65-69 YO','70-74 YO','75-79 YO','Over 80 YO'],
            key='X_AGEG5YR'
        )
        X_RACE = st.selectbox(
            label='Race/Ethnicity:',
            options=['White', 'Black','American Indian or Alaskan Native','Asian','Native Hawaiian or other Pacific Islander','Other race','Multirace','Hispanic'],
            key='X_RACE'
        )
        X_EDUCAG = st.selectbox(
            label='What is your education level?',
            options=['Did not graduate high school', 'Graduated high school', 'Attended college','Graduated college'],
            key='X_EDUCAG'
        )
        EMPLOY1 = st.selectbox(
            label='What is your employment status?',
            options=['Employed', 'Self employed','No work (over an year)','No work (less than an year)','Homemaker','Student','Retired','Unable to work'],
            key='EMPLOY1'
        )
        INCOME3 = st.selectbox(
            label='What is your annual income?',
            options=['Less than 10K', '10-15K','15-20K','20-25K','25-35K','35-50K','50-75K','75-100K','100-150K','150-200K','Over 200K'],
            key='INCOME3'
        )
        MARITAL = st.selectbox(
            label='Marital status',
            options=['Married', 'Divorced','Widowed','Separated','Never Married','Unmarried Couple'],
            key='MARITAL'
        )
        RENTHOM1 = st.selectbox(
            label='Home owner status',
            options=['Own home', 'Rent home','Other home options'],
            key='RENTHOM1'
        )
        PRIMINSR = st.selectbox(
            label='What is the primary source of your health care coverage?',
            options=['Employer paid Insurance', 'Self paid Insurance', 'Medicare Insurance', 'Medigap Insurance', 'Medicaid Insurance', 'CHIP Insurance','VA/Military Insurance','Indian Health Service', 'State sponsored health plan', 'Other Insurance', 'No Insurance'],
            key='PRIMINSR'
        )
        CHECKUP1 = st.selectbox(
            label='how long has it been since you last visited a doctor for a routine checkup?',
            options=['Last checkup was less than a year ago', 'Last checkup was less than 1-2 years ago', 'Last checkup was less than 3-5 years ago','Last checkup was  Over 5 years ago','Never had a checkup'],
            key='CHECKUP1'
        )
        GENHLTH = st.selectbox(
            label='Would you say that in general your health is:',
            options=['Excellent Health', 'Very Good Health', 'Good Health','Fair Health','Poor Health'],
            key='GENHLTH'
        )
        PHYSHLTH14D = st.selectbox(
            label='Over the last month, how many days has your physical health not been good (due to injury or illness)?',
            options=['0 Days of bad physical health', '1-13 Days of bad physical health', 'Over 14 Days of bad physical health'],
            key='PHYSHLTH14D'
        )
        MENTHLTH14D = st.selectbox(
            label='Over the last month, how many days has your mental health not been good (due to stress or depression or other reasons)?',
            options=['0 Days of bad mental health', '1-13 Days of bad mental health', 'Over 14 Days of bad mental health'],
            key='MENTHLTH14D'
        )
        EXERANY2 = st.selectbox(
            label='In the past month, did you participate in any physical activities or exercises outside of work?',
            options=['Exercised', 'Did not exercise'],
            key='EXERANY2'
        )
        X_FRUTSU1DF = st.selectbox(
            label='How many fruit and fruit products consumed per day?',
            options=['Less than once a day (Fruit Consumption)', 'Less than twice a day (Fruit Consumption)', 'Less than 5 times a day (Fruit Consumption)','Over 5 times a day (Fruit Consumption)'],
            key='X_FRUTSU1DF'
        )
        X_VEGSU1DF = st.selectbox(
            label='How many green vegetables and other vegetables (excluding potatoes) consumed per day?',
            options=['Less than once a day (Vegetable Consumption)', 'Less than twice a day (Vegetable Consumption)', 'Less than 5 times a day (Vegetable Consumption)','Over 5 times a day (Vegetable Consumption)'],
            key='X_VEGSU1DF'
        )
        VACCSTAT = st.selectbox(
            label='Have you had the flu and pneumonia vaccine?',
            options=['Both Flu and Pneumonia vaccines', 'Pneumonia vaccine only', 'Flu vaccine only', 'Not vaccinated for either'],
            key='VACCSTAT'
        )
    
    with (lg_col2):
            # Maximum iterations to run the LG until convergence
        ALCOFREQ = st.selectbox(
            label='How many days in the last 30 days have you consumed alcohol?',
            options=['Dont drink (0 days)', 'Occasional drinker (1-7 days)','Frequent drinker (7-14 days)', 'Regular drinker (> 15 days)'],
            key='ALCOFREQ'
            )
        X_SMOKE = st.selectbox(
            label='What is your smoking status?',
            options=['Current Smoker (Daily)', 'Current Smoker (Some days)','Former Smoker', 'Non Smoker'],
            key='X_SMOKE'
        )
        BPHIGH6 = st.selectbox(
            label='Ever been told that you have high blood pressure?',
            options=['Had high BP', 'Had high BP (Pregnant)', 'Did not have high BP','Borderline high BP'],
            key='BPHIGH6'
        )
        CHOLSTAT = st.selectbox(
            label='Cholesterol Status:',
            options=['High Cholesterol and take medicine', 'High Cholesterol but do not take Medicine', 'Low Cholesterol and Take medicine','Low Cholesterol and do not take medicine'],
            key='CHOLSTAT'
        )
        X_BMI5CAT = st.selectbox(
            label='Based on your BMI, are you?',
            options=['Under weight', 'Normal weight', 'Over weight', 'Obese'],
            key='X_BMI5CAT'
        )
        X_ASTHMS1 = st.selectbox(
            label='Do you or have you had asthma?',
            options=['Have asthma currently', 'Had asthma before', 'Never have had asthma'],
            key='X_ASTHMS1'
        )
        CVDINFR4 = st.selectbox(
            label='Ever been told that you have had a heart attack or myocardial infarction?',
            options=['Had a heart attack', 'Did not have a heart attack'],
            key='CVDINFR4'
        )
        CVDCRHD4 = st.selectbox(
            label='Ever been told that you have had coronary heart disease or angina?',
            options=['Had coronary heart disease', 'Did not have coronary heart disease'],
            key='CVDCRHD4'
        )
        CVDSTRK3 = st.selectbox(
            label='Ever been told that you have had a stroke?',
            options=['Had a stroke', 'Did not have a stroke'],
            key='CVDSTRK3'
        )
        CHCCOPD3 = st.selectbox(
            label='Ever been told that you have C.O.P.D.(chronic obstructive pulmonary disease), emphysema or chronic bronchitis?',
            options=['Had COPD', 'Did not have COPD'],
            key='CHCCOPD3'
        )
        ADDEPEV3 = st.selectbox(
            label='Ever been told that you have a depressive disorder (including depression, major depression, dysthymia, or minor depression)?',
            options=['Had depressive order', 'Did not have depressive order'],
            key='ADDEPEV3'
        )
        CHCKDNY2 = st.selectbox(
            label='Ever been told that you have kidney disease (excluding kidney stones, bladder infection and incontinence)?',
            options=['Had kidney disease', 'Did not have kidney disease'],
            key='CHCKDNY2'
        )
        HAVARTH5 = st.selectbox(
            label='Ever been told that you have arthritis?',
            options=['Had arthritis', 'Did not have arthritis'],
            key='HAVARTH5'
        )
        CHCOCNCR = st.selectbox(
            label='Ever been told that you have cancer?',
            options=['Had cancer', 'Did not have cancer'],
            key='CHCOCNCR'
        )
        BLIND = st.selectbox(
            label='Do you have serious difficulty seeing, even when wearing glasses?',
            options=['Difficulty seeing', 'No difficulty seeing'],
            key='BLIND'
        )
        DECIDE = st.selectbox(
            label='Do you have serious difficulty concentrating, remembering, or making decisions?',
            options=['Difficulty deciding', 'No difficulty deciding'],
            key='DECIDE'
        )
    
    input_row = {'SEXVAR':SEXVAR,
                    'GENHLTH':GENHLTH,
                    'PRIMINSR':PRIMINSR,
                    'CHECKUP1':CHECKUP1,
                    'EXERANY2':EXERANY2,
                     'BPHIGH6':BPHIGH6,
                     'CVDINFR4':CVDINFR4,
                     'CVDCRHD4':CVDCRHD4,
                     'CVDSTRK3':CVDSTRK3,
                     'CHCCOPD3':CHCCOPD3,
                     'ADDEPEV3':ADDEPEV3,
                     'CHCKDNY2':CHCKDNY2,
                     'HAVARTH5':HAVARTH5,
                     'MARITAL':MARITAL,
                     'RENTHOM1':RENTHOM1,
                     'EMPLOY1':EMPLOY1,
                     'INCOME3':INCOME3,
                     'BLIND':BLIND,
                     'DECIDE':DECIDE,
                     'X_AGEG5YR':X_AGEG5YR,
                     'X_RACE':X_RACE,
                     'X_EDUCAG':X_EDUCAG,
                     'X_SMOKE':X_SMOKE,
                     'X_ASTHMS1':X_ASTHMS1,
                     'X_BMI5CAT':X_BMI5CAT,
                     'CHCOCNCR':CHCOCNCR,
                     'PHYSHLTH14D':PHYSHLTH14D,
                     'MENTHLTH14D':MENTHLTH14D,
                     'ALCOFREQ':ALCOFREQ,
                     'VACCSTAT':VACCSTAT,
                     'CHOLSTAT':CHOLSTAT,
                     'X_FRUTSU1DF':X_FRUTSU1DF,
                     'X_VEGSU1DF':X_VEGSU1DF
                    }   
    
    #st.write(input_row)
    
    #st.write(X_predict)
    ###################### VISUALIZE DATASET #######################
         
    ###################### VISUALIZE DATASET #######################
    ## The models were trained using the following parameters
    #feature_input_select = df.columns.drop('DIABETERES')
    #X_train, X_test, y_train, y_test = split_dataset(df, 30, 'DIABETERES', feature_input_select, sample_opt=4,oversample_val = 25/100,undersample_val=50/100)
    #lr_sgd = LogisticRegression_SGD(num_iterations=7500, learning_rate=0.0005, batch_size = 7500)
    #neigh = KNeighborsClassifier(weights='distance', p=2,leaf_size=10,n_neighbors=100)
    #clf = RandomForestClassifier(criterion='entropy', min_samples_leaf = 20,max_depth=50,n_estimators=1000)
    #logmodel = LogisticRegression(max_iter = 10000, solver = 'saga', C=0.1, tol=0.001)
    #treed = DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, max_depth=10)
    #lsvm = LinearSVC(dual=False, random_state=0,max_iter=1000,tol=0.05)
    ## For training, we used the oversampled/undersampled mix data from the initial broken down 70:30 split
    with open(r"./Models/lr_sgd.pkl", "rb") as input_file:
            lr_sgd_mod = pickle.load(input_file)
    with open(r"./Models/rlr.pkl", "rb") as input_file:
            rlr_mod = pickle.load(input_file)
    with open(r"./Models/knn.pkl", "rb") as input_file:
            knn_mod = pickle.load(input_file)
    with open(r"./Models/rf.pkl", "rb") as input_file:
            rf_mod = pickle.load(input_file)
    with open(r"./Models/lsvm.pkl", "rb") as input_file:
            lsvm_mod = pickle.load(input_file)
    with open(r"./Models/lsvm_w.pkl", "rb") as input_file:
            lsvmw_mod = pickle.load(input_file)
    with open(r"./Models/rf_w.pkl", "rb") as input_file:
            rfw_mod = pickle.load(input_file)
    with open(r"./Models/rlr_w.pkl", "rb") as input_file:
            rlrw_mod = pickle.load(input_file)
    
    if st.button("Predict Results"):
        X_predict = split_dataset_predict(df, 0.3, input_row,sample_opt=4,oversample_val=0.25, undersample_val=0.5,random_state=42)
        lr_sgd = lr_sgd_mod.predict(X_predict)
        lr_sgd = lr_sgd[0]
        rlr = rlr_mod.predict(X_predict)
        knn = knn_mod.predict(X_predict)
        rf = rf_mod.predict(X_predict)
        lsvm = lsvm_mod.predict(X_predict)
        rlrw = rlrw_mod.predict(X_predict)
        rfw = rfw_mod.predict(X_predict)
        lsvmw = lsvmw_mod.predict(X_predict)
        y_pred_uw = [lr_sgd,rlr,knn,rf,lsvm]
        y_pred_w = [rlrw,rfw,lsvmw]
        u1=len([i for j,i in enumerate(y_pred_uw) if i==1])
        u0=len([i for j,i in enumerate(y_pred_uw) if i==0])
        w1=len([i for j,i in enumerate(y_pred_w) if i==1])
        w0=len([i for j,i in enumerate(y_pred_w) if i==0])

        st.markdown('#### {} Unweighted Models predict you have diabetes while {} models predict that you do not have diabetes:'.format(u1,u0))

        if lr_sgd == 0:
            st.write('The Logistic Regression using Stochastic Gradient Descent model predicts that you do not stand at the risk of having Diabetes.')
        if lr_sgd == 1:
            st.write('The Logistic Regression using Stochastic Gradient Descent model predicts you do stand at the risk of having Diabetes.')

        if rlr == 0:
            st.write('The Regularized Logistic Regression model predicts that you do not stand at the risk of having Diabetes.')
        if rlr == 1:
            st.write('The Regularized Logistic Regression model predicts you do stand at the risk of having Diabetes.')

        if knn == 0:
            st.write('The K Nearest Neighbor model predicts that you do not stand at the risk of having Diabetes.')
        if knn == 1:
            st.write('The K Nearest Neighbor model predicts you do stand at the risk of having Diabetes.')

        if rf == 0:
            st.write('The Random Forest classifier model predicts that you do not stand at the risk of having Diabetes.')
        if rf == 1:
            st.write('The Random Forest classifier model predicts you do stand at the risk of having Diabetes.')

        if lsvm == 0:
            st.write('The Linear Support Vector Machine model predicts that you do not stand at the risk of having Diabetes.')
        if lsvm == 1:
            st.write('The Linear Support Vector Machine model predicts you do stand at the risk of having Diabetes.')
        st.markdown("""
        """)
        st.markdown('#### {} Weighted Models predict you have diabetes while {} models predict that you do not have diabetes:'.format(w1,w0))
        if rlrw == 0:
            st.write('The Regularized Logistic Regression weighted model predicts that you do not stand at the risk of having Diabetes.')
        if rlrw == 1:
            st.write('The Regularized Logistic Regression weighted model predicts you do stand at the risk of having Diabetes.')

        if rfw == 0:
            st.write('The Random Forest classifier weighted model predicts that you do not stand at the risk of having Diabetes.')
        if rfw == 1:
            st.write('The Random Forest classifier weighted model predicts you do stand at the risk of having Diabetes.')

        if lsvmw == 0:
            st.write('The Linear Support Vector Machine weighted model predicts that you do not stand at the risk of having Diabetes.')
        if lsvmw == 1:
            st.write('The Linear Support Vector Machine weighted model predicts you do stand at the risk of having Diabetes.')

        st.write("Please refer to the **Explore Results** page regarding how well the model works.")


        
        
        
        
