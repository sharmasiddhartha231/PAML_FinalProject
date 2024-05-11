import numpy as np                    
from sklearn.model_selection import train_test_split
import streamlit as st                  
import random
from helper_functions import fetch_dataset, set_pos_neg_reviews
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
import seaborn as sns
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
# set seed=10 to produce consistent results
random.seed(10)

#############################################

st.title('Model Exploration')

#############################################

st.markdown("""Welcome to the **Model Exploration** section where you can test the various machine learning models we have built by yourself using any selection of parameters and see they perform. The following machine learning models are available for you to work with:
- Logistic Regression using Gradient Descent
- Logistic Regression using Stochastic Gradient Descent
- Regularized Logistic Regression 
- K Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest Classifier
- Support Vector Machines
""")

def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    df = None
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        filepath = "/Users/siddharthasharma/Desktop/PAML/PAML_FinalProject/Diabetes_Data_Sub_Strict_Main_String_New.txt"
        df = pd.read_csv(filepath, sep='\t')
    if df is not None:
        st.session_state['data'] = df
    return df

# Checkpoint 4
def split_dataset(df, number, target, input_var, sample_opt=1,oversample_val=0.25, undersample_val=0.5,random_state=42):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    X_train_main = []
    X_test_main = []
    y_train_main = []
    y_test_main = []
    
    # Add code here
    df = df.drop(df[df.DIABETERES == 'Prediabetes'].index)
    df.DIABETERES[df.DIABETERES == 'No Diabetes'] = 0
    df.DIABETERES[df.DIABETERES == 'Diabetes'] = 1
    df = df.reset_index(drop=True)
    X, y = df.loc[:, df.columns.isin(input_var)], df.loc[:, df.columns.isin([target])]
    #enc = OneHotEncoder(handle_unknown='ignore')
    #enc.fit(X)
    for i in input_var:
        i = pd.get_dummies(X[i], drop_first=False)
        X = pd.concat([X,i], axis=1)
    X = X.loc[:, ~X.columns.isin(input_var)]
    X.columns = X.columns.astype(str)
    X = X.replace(False,0, regex=True)
    X = X.replace(True,1, regex=True)
    y=y.astype('int')
    over = SMOTE(sampling_strategy=oversample_val)
    under = RandomUnderSampler(sampling_strategy=undersample_val)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(X, y, test_size=number/100, random_state=random_state)
    if sample_opt == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number/100, random_state=random_state)    
    if sample_opt == 2:
        X,y = over.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number/100, random_state=random_state)
    if sample_opt == 3:
        X,y = under.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number/100, random_state=random_state)
    if sample_opt == 4:
        X,y = pipeline.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number/100, random_state=random_state)

    return X_train, X_test_main, y_train, y_test_main

def compute_evaluation(prediction_labels, true_labels, estimator_name):    
    '''
    Compute classification accuracy
    Input
        - prediction_labels (numpy): predicted product sentiment
        - true_labels (numpy): true product sentiment
    Output
        - accuracy (float): accuracy percentage (0-100%)
    '''
    precision=0
    misclassification=0
    accuracy=0
    sensitivity=0
    specificity=0
    auc_roc = 0
    # Add code here
    metric_dict = {'Precision': -1,
                   'Sensitivity': -1,
                   'Accuracy': -1,
                   'Specificity': -1,
                   'Misclassification':-1,
                   'Area under the ROC curve':-1}
    cmatrix = confusion_matrix(true_labels, prediction_labels, labels=[0,1])
    tn, fp, fn, tp = cmatrix.ravel()
    specificity = tn / (tn+fp)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    misclassification = (fp + fn)/(tp + tn + fp + fn)
    sensitivity = tp /(tp + fn)
    precision = tp / (tp + fp)
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, prediction_labels)
    auc_roc= metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_roc,estimator_name=estimator_name)

    metric_dict['Precision'] = precision
    metric_dict['Sensitivity'] = sensitivity
    metric_dict['Accuracy'] = accuracy
    metric_dict['Specificity'] = specificity
    metric_dict['Misclassification'] = misclassification
    metric_dict['Area under the ROC curve'] = auc_roc
    return metric_dict, display

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
        y1 = (Y * np.log(score))
        y2 = (1-Y) * np.log(1 - score)
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

##Tried using Weighted Logistic Regression to calculate odds ratio but did not work correctly with the Gradient Descent Method so ended up using the sklearn Logistic Regression Methodology.
#class WeightedLogisticRegression(object):
#    def __init__(self, learning_rate=0.001, num_iterations=1000): 
#        self.learning_rate = learning_rate 
#        self.num_iterations = num_iterations 
#        self.likelihood_history=[]
#    def predict_probability(self, X):
#        score = np.dot(X, self.W) + self.b
#        y_pred = 1. / (1.+np.exp(-score)) 
#        return y_pred
#    def compute_avg_log_likelihood(self, X, Y, W):
        #indicator = (Y==+1)
        #scores = np.dot(X, W) 
        #logexp = np.log(1. + np.exp(-scores))
        #mask = np.isinf(logexp)
        #logexp[mask] = -scores[mask]
        #lp = np.sum((indicator-1)*scores - logexp)/len(X)
#        scores = np.dot(X, W) 
#        score = 1 / (1 + np.exp(-scores))
#        y1 = 3.33 *(Y * np.log(score))
#        y2 = 0.59 * (1-Y) * np.log(1 - score)
#        lp = -np.mean(y1 + y2)
#        return lp
#    def update_weights(self):      
#        num_examples, num_features = self.X.shape
#        y_pred = self.predict(self.X)
#        dW = self.X.T.dot(self.Y-y_pred) / num_examples 
#        db = np.sum(self.Y-y_pred) / num_examples 
#        self.b = self.b + self.learning_rate * db
#        self.W = self.W + self.learning_rate * dW
#        log_likelihood=0
#        log_likelihood += self.compute_avg_log_likelihood(self.X, self.Y, self.W)
#        self.likelihood_history.append(log_likelihood)
#    def predict(self, X):
#        y_pred= 0
#        scores = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
#        y_pred = [0 if z <= 0.5 else +1 for z in scores]
#        return y_pred 
#    def fit(self, X, Y):   
#        self.X = X
#        self.Y = Y
#        num_examples, num_features = self.X.shape    
#        self.W = np.zeros(num_features)
#        self.b = 0
#        self.likelihood_history=[]
#        for _ in range(self.num_iterations):          
#            self.update_weights()  
#    def get_weights(self):
#            out_dict = {'Logistic Regression': []}
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
df = fetch_dataset()

if df is not None:
    feature_predict_select = 'DIABETERES'
    # Select input features
    feature_input_select = st.multiselect(
        label='Select features for classification input',
        options=[f for f in list(df.columns) if f != feature_predict_select],
        key='feature_select'
    )

    st.session_state['feature'] = feature_input_select

    st.write('You selected input {}'.format(
        feature_input_select))

    sample_select = st.selectbox(
        label='Do you wish to oversample/undersample the dataset',
        options=['No', 'Oversample', 'Undersample', 'Oversample minority and undersample majority'],
        key='sample_select'
    )
    st.session_state['oversample'] = sample_select
    if sample_select == 'Oversample':
        oversampling_rate = st.number_input(
                label='Input oversampling ratio',
                min_value=25,
                max_value=100,
                value=25,
                step=1,
                key='oversampling_rate'
            )
    if sample_select == 'Undersample':
        undersampling_rate = st.number_input(
                label='Input undersampling ratio',
                min_value=25,
                max_value=100,
                value=50,
                step=1,
                key='undersampling_rate'
            )
    if sample_select == 'Oversample minority and undersample majority':
        oversampling_rate = st.number_input(
                label='Input oversampling ratio',
                min_value=25,
                max_value=100,
                value=25,
                step=1,
                key='oversampling_rate'
            )
        undersampling_rate = st.number_input(
                label='Input undersampling ratio',
                min_value=25,
                max_value=100,
                value=50,
                step=1,
                key='undersampling_rate'
            )

    st.session_state['oversample'] = sample_select
    # Task 4: Split train/test
    st.markdown('### Split dataset into Train/Test sets')
    st.markdown(
        '#### Enter the percentage of training data to use for training the model')
    number = st.number_input(
        label='Enter size of train set (X%)', min_value=0, max_value=100, value=30, step=1)
    number = 100 - number
    X_train, X_test, y_train, y_test = [], [], [], []
    # Compute the percentage of test and training data
    if (feature_predict_select in df.columns and sample_select == 'No'):
        X_train, X_test, y_train, y_test = split_dataset(
            df, number, feature_predict_select, feature_input_select, sample_opt=1)
    if (feature_predict_select in df.columns and sample_select == 'Oversample'):
        X_train, X_test, y_train, y_test = split_dataset(
            df, number, feature_predict_select, feature_input_select, sample_opt=2,oversample_val = oversampling_rate/100)
    if (feature_predict_select in df.columns and sample_select == 'Undersample'):
        X_train, X_test, y_train, y_test = split_dataset(
            df, number, feature_predict_select, feature_input_select, sample_opt=3,undersample_val=undersampling_rate/100)
    if (feature_predict_select in df.columns and sample_select == 'Oversample minority and undersample majority'):
        X_train, X_test, y_train, y_test = split_dataset(
            df, number, feature_predict_select, feature_input_select, sample_opt=4,oversample_val = oversampling_rate/100,undersample_val=undersampling_rate/100)
    st.write('Number of entries in training set: {}'.format(X_train.shape[0]))
    st.write('Number of entries in testing set: {}'.format(X_test.shape[0]))

    classification_methods_options = ['Logistic Regression using Gradient Descent',
                                      'Logistic Regression using Stochastic Gradient Descent'
                                      'Regularized Logistic Regression',
                                      'K Nearest Neighbor',
                                      'Decision Tree',
                                      'Random Forest']

    trained_models = [
        model for model in classification_methods_options if model in st.session_state]

    st.session_state['trained_models'] = trained_models
    
    # Collect ML Models of interests
    classification_model_select = st.selectbox(
        label='Select classification model for prediction',
        options=classification_methods_options,
    )
    st.write('You selected the follow models: {}'.format(classification_model_select))

    # Add parameter options to each regression method
    # Task 5: Logistic Regression
    if (classification_methods_options[0] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[0])

        lg_col1, lg_col2 = st.columns(2)

        with (lg_col1):
            lg_learning_rate_input = st.text_input(
                label='Input learning rate 👇',
                value='0.001',
                key='lg_learning_rate_textinput'
            )
            st.write('You select the following learning rate value(s): {}'.format(lg_learning_rate_input))

        with (lg_col2):
            # Maximum iterations to run the LG until convergence
            lg_num_iterations = st.number_input(
                label='Enter the number of maximum iterations on training data',
                min_value=1000,
                max_value=25000,
                value=1000,
                step=100,
                key='lg_max_iter_numberinput'
            )
            st.write('You set the maximum iterations to: {}'.format(lg_num_iterations))

        lg_params = {
            'num_iterations': lg_num_iterations,
            'learning_rate': [float(val) for val in lg_learning_rate_input.split(',')],
        }
        lg_model = LogisticRegression_GD(num_iterations=lg_params['num_iterations'], learning_rate=lg_params['learning_rate'][0])
        lg_model.fit(X_train.to_numpy(), np.ravel(y_train)) 
        st.write('Logistic Regression Model using Gradient Descent trained')
        y_pred = lg_model.predict(X_test)
        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification','Area under the ROC curve'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        
        Metric_data, ROC_Curve = compute_evaluation(y_pred, y_test, classification_methods_options[0])
        if 'Accuracy' in evaluation_metric_select:
           st.write('Accuracy of the current model is: {}'.format(Metric_data['Accuracy'])) 
        if 'Precision' in evaluation_metric_select:
           st.write('Precision of the current model is: {}'.format(Metric_data['Precision'])) 
        if 'Sensitivity' in evaluation_metric_select:
           st.write('Sensitivity of the current model is: {}'.format(Metric_data['Sensitivity'])) 
        if 'Specificity' in evaluation_metric_select:
           st.write('Specificity of the current model is: {}'.format(Metric_data['Specificity'])) 
        if 'Misclassification' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Misclassification']))
        if 'Area under the ROC curve' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Area under the ROC curve']))
        plot_curve_select = st.selectbox(
        label='Plot ROC curve',
        options=['No', 'Yes'],
        )
        if plot_curve_select == 'Yes':
           ROC_Curve.plot()
           RC = plt.show()
           st.set_option('deprecation.showPyplotGlobalUse', False)
           st.pyplot(RC)

if (classification_methods_options[1] in classification_model_select):
        st.markdown('#### ' + classification_methods_options[1])

        # Number of iterations: maximum iterations to run the iterative SGD
        sdg_num_iterations = st.number_input(
            label='Enter the number of maximum iterations on training data',
            min_value=1000,
            max_value=50000,
            value=500,
            step=100,
            key='sgd_num_iterations_numberinput'
        )
        st.write('You set the maximum iterations to: {}'.format(sdg_num_iterations))

        # learning_rate: Constant that multiplies the regularization term. Ranges from [0 Inf)
        sdg_learning_rate = st.text_input(
            label='Input one alpha value',
            value='0.001',
            key='sdg_learning_rate_numberinput'
        )
        sdg_learning_rate = float(sdg_learning_rate)
        st.write('You selected the following learning rate: {}'.format(sdg_learning_rate))

        # tolerance: stopping criteria for iterations
        sgd_batch_size = st.text_input(
            label='Input a batch size value',
            value='500',
            key='sgd_batch_size_textinput'
        )
        sgd_batch_size = int(sgd_batch_size)
        st.write('You selected the following batch_size: {}'.format(sgd_batch_size))

        sgd_params = {
            'num_iterations': sdg_num_iterations,
            'batch_size': sgd_batch_size,
            'learning_rate': sdg_learning_rate,
        }

        lg_model = LogisticRegression_SGD(num_iterations=sgd_params['num_iterations'], learning_rate=sgd_params['learning_rate'],batch_size=sgd_params['batch_size'])
        lg_model.fit(X_train.to_numpy(), np.ravel(y_train)) 
        st.write('Logistic Regression Model using Stochastic Gradient Descent trained')
        y_pred = lg_model.predict(X_test)
        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification','Area under the ROC curve'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        Metric_data, ROC_Curve = compute_evaluation(y_pred, y_test, classification_methods_options[2])
        if 'Accuracy' in evaluation_metric_select:
           st.write('Accuracy of the current model is: {}'.format(Metric_data['Accuracy'])) 
        if 'Precision' in evaluation_metric_select:
           st.write('Precision of the current model is: {}'.format(Metric_data['Precision'])) 
        if 'Sensitivity' in evaluation_metric_select:
           st.write('Sensitivity of the current model is: {}'.format(Metric_data['Sensitivity'])) 
        if 'Specificity' in evaluation_metric_select:
           st.write('Specificity of the current model is: {}'.format(Metric_data['Specificity'])) 
        if 'Misclassification' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Misclassification']))
        if 'Area under the ROC curve' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Area under the ROC curve']))
        plot_curve_select = st.selectbox(
        label='Plot ROC curve',
        options=['No', 'Yes'],
        )
        if plot_curve_select == 'Yes':
           ROC_Curve.plot()
           RC = plt.show()
           st.set_option('deprecation.showPyplotGlobalUse', False)
           st.pyplot(RC)


if (classification_methods_options[2] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[2])
        # Number of iterations: maximum iterations to run the iterative SGD
        rlr_num_iterations = st.number_input(
            label='Enter the number of maximum iterations on training data',
            min_value=1000,
            max_value=100000,
            value=1000,
            step=1000,
            key='rlr_num_iterations_numberinput'
        )
        st.write('You set the maximum iterations to: {}'.format(rlr_num_iterations))

        # learning_rate: Constant that multiplies the regularization term. Ranges from [0 Inf)
        rlr_tolerance = st.text_input(
            label='Input tolerance value',
            value='0.0001',
            key='rlr_tolerance'
        )
        rlr_tolerance = float(rlr_tolerance)
        st.write('You selected the following learning rate: {}'.format(rlr_tolerance))

        # tolerance: stopping criteria for iterations
        rlr_regularization_value = st.text_input(
            label='Input a regularization value',
            value='1',
            key='rlr_regularization_value'
        )
        rlr_regularization_value = float(rlr_regularization_value)
        st.write('You selected the following batch_size: {}'.format(rlr_regularization_value))

        rlr_solver = st.selectbox(
            label='Input solver name',
            options=['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'],
            key='rlr_solver'
        )
        st.write('You selected the following batch_size: {}'.format(rlr_solver))

        rlr_params = {
            'num_iterations': rlr_num_iterations,
            'tolerance': rlr_tolerance,
            'solver': rlr_solver,
            'regularization': rlr_regularization_value
        }
        lg_model = LogisticRegression(max_iter = sdg_num_iterations,solver=rlr_solver,C=rlr_regularization_value, tol=rlr_tolerance)
        lg_model.fit(X_train, y_train)
        st.write('Regularized Logistic Regression Model Model trained')
        y_pred = lg_model.predict(X_test)
        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification','Area under the ROC curve'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        Metric_data, ROC_Curve = compute_evaluation(y_pred, y_test, classification_methods_options[2])
        if 'Accuracy' in evaluation_metric_select:
           st.write('Accuracy of the current model is: {}'.format(Metric_data['Accuracy'])) 
        if 'Precision' in evaluation_metric_select:
           st.write('Precision of the current model is: {}'.format(Metric_data['Precision'])) 
        if 'Sensitivity' in evaluation_metric_select:
           st.write('Sensitivity of the current model is: {}'.format(Metric_data['Sensitivity'])) 
        if 'Specificity' in evaluation_metric_select:
           st.write('Specificity of the current model is: {}'.format(Metric_data['Specificity'])) 
        if 'Misclassification' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Misclassification']))
        if 'Area under the ROC curve' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Area under the ROC curve']))
        plot_curve_select = st.selectbox(
        label='Plot ROC curve',
        options=['No', 'Yes'],
        )
        if plot_curve_select == 'Yes':
           ROC_Curve.plot()
           RC = plt.show()
           st.set_option('deprecation.showPyplotGlobalUse', False)
           st.pyplot(RC)

if (classification_methods_options[3] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[3])
        lg_model = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
        lg_model.fit(X_train, y_train)
        st.write('K Nearest Neighbor Model trained')
        y_pred = lg_model.predict(X_test)
        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification'] 
        evaluation_metric_select = st.multiselect(
            label='Select evaluation metric for current model',
            options=evaluation_options,
            key='evaluation_select'
            )
        st.session_state['evaluation'] = evaluation_metric_select
        Metric_data, ROC_Curve = compute_evaluation(y_pred, y_test, classification_methods_options[3])
        if 'Accuracy' in evaluation_metric_select:
            st.write('Accuracy of the current model is: {}'.format(Metric_data['Accuracy'])) 
        if 'Precision' in evaluation_metric_select:
            st.write('Precision of the current model is: {}'.format(Metric_data['Precision'])) 
        if 'Sensitivity' in evaluation_metric_select:
            st.write('Sensitivity of the current model is: {}'.format(Metric_data['Sensitivity'])) 
        if 'Specificity' in evaluation_metric_select:
            st.write('Specificity of the current model is: {}'.format(Metric_data['Specificity'])) 
        if 'Misclassification' in evaluation_metric_select:
            st.write('Misclassification of the current model is: {}'.format(Metric_data['Misclassification']))
        if 'Area under the ROC curve' in evaluation_metric_select:
            st.write('Misclassification of the current model is: {}'.format(Metric_data['Area under the ROC curve']))
        plot_curve_select = st.selectbox(
        label='Plot ROC curve',
        options=['No', 'Yes'],
        )
        if plot_curve_select == 'Yes':
           ROC_Curve.plot()
           RC = plt.show()
           st.set_option('deprecation.showPyplotGlobalUse', False)
           st.pyplot(RC)

if (classification_methods_options[4] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[4])
        lg_model = DecisionTreeClassifier()
        lg_model.fit(X_train, y_train)
        st.write('Decision Tree Model trained')
        y_pred = lg_model.predict(X_test)
        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        Metric_data, ROC_Curve = compute_evaluation(y_pred, y_test, classification_methods_options[4])
        if 'Accuracy' in evaluation_metric_select:
           st.write('Accuracy of the current model is: {}'.format(Metric_data['Accuracy'])) 
        if 'Precision' in evaluation_metric_select:
           st.write('Precision of the current model is: {}'.format(Metric_data['Precision'])) 
        if 'Sensitivity' in evaluation_metric_select:
           st.write('Sensitivity of the current model is: {}'.format(Metric_data['Sensitivity'])) 
        if 'Specificity' in evaluation_metric_select:
           st.write('Specificity of the current model is: {}'.format(Metric_data['Specificity'])) 
        if 'Misclassification' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Misclassification'])) 
        if 'Area under the ROC curve' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Area under the ROC curve']))
        plot_curve_select = st.selectbox(
        label='Plot ROC curve',
        options=['No', 'Yes'],
        )
        if plot_curve_select == 'Yes':
           ROC_Curve.plot()
           RC = plt.show()
           st.set_option('deprecation.showPyplotGlobalUse', False)
           st.pyplot(RC)

if (classification_methods_options[5] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[5])
        lg_model = RandomForestClassifier(random_state=0)
        lg_model.fit(X_train, y_train)
        st.write('Random Forest Model trained')
        y_pred = lg_model.predict(X_test)
        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        Metric_data, ROC_Curve = compute_evaluation(y_pred, y_test, classification_methods_options[5])
        if 'Accuracy' in evaluation_metric_select:
           st.write('Accuracy of the current model is: {}'.format(Metric_data['Accuracy'])) 
        if 'Precision' in evaluation_metric_select:
           st.write('Precision of the current model is: {}'.format(Metric_data['Precision'])) 
        if 'Sensitivity' in evaluation_metric_select:
           st.write('Sensitivity of the current model is: {}'.format(Metric_data['Sensitivity'])) 
        if 'Specificity' in evaluation_metric_select:
           st.write('Specificity of the current model is: {}'.format(Metric_data['Specificity'])) 
        if 'Misclassification' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Misclassification'])) 
        if 'Area under the ROC curve' in evaluation_metric_select:
           st.write('Misclassification of the current model is: {}'.format(Metric_data['Area under the ROC curve']))
        plot_curve_select = st.selectbox(
        label='Plot ROC curve',
        options=['No', 'Yes'],
        )
        if plot_curve_select == 'Yes':
           ROC_Curve.plot()
           RC = plt.show()
           st.set_option('deprecation.showPyplotGlobalUse', False)
           st.pyplot(RC)
