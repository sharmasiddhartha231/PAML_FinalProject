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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
# set seed=10 to produce consistent results
random.seed(10)

#############################################

st.title('Model Exploration')

#############################################

st.markdown("""Welcome to the **Model Exploration** section where you can test the various machine learning models we have built by yourself using any selection of parameters and see they perform. The following machine learning models are available for you to work with:
- Logistic Regression
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
    data = None
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        data = "/Users/siddharthasharma/Desktop/PAML/PAML_FinalProject/Diabetes_Data_Sub_Strict_Main_String_New.txt"
        df = pd.read_csv(data, sep='\t')
    if df is not None:
        st.session_state['diabetes'] = df
    return df

# Checkpoint 4
def split_dataset(df, number, target, input_var, oversample=False,random_state=42):
    """
    This function splits the dataset into the training and test sets.

    Input:
        - X: training features
        - y: training targets
        - number: the ratio of test samples
        - target: article feature name 'rating'
        - feature_encoding: (string) 'Word Count' or 'TF-IDF' encoding
        - random_state: determines random number generation for centroid initialization
    Output:
        - X_train_sentiment: training features (word encoded)
        - X_val_sentiment: test/validation features (word encoded)
        - y_train: training targets
        - y_val: test/validation targets
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
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

    if oversample == True:
        over = SMOTE(sampling_strategy=1)
        # transform the dataset
        X, y = over.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number/100, random_state=random_state)
    return X_train, X_test, y_train, y_test

def compute_evaluation(prediction_labels, true_labels):    
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
    auc_roc_curve = 0
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
    auc_roc_curve = metrics.auc(fpr, tpr)

    metric_dict['Precision'] = precision
    metric_dict['Sensitivity'] = sensitivity
    metric_dict['Accuracy'] = accuracy
    metric_dict['Specificity'] = specificity
    metric_dict['Misclassification'] = misclassification
    metric_dict['Area under the ROC curve'] = auc_roc_curve
    return metric_dict

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

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:

    # Display dataframe as table
    #st.dataframe(df)
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

    oversample_select = st.selectbox(
        label='Do you wish to oversample the dataset (Highly Recommended)',
        options=['No', 'Yes'],
        key='oversample_select'
    )

    st.session_state['oversample'] = oversample_select
    # Task 4: Split train/test
    st.markdown('### Split dataset into Train/Test sets')
    st.markdown(
        '#### Enter the percentage of training data to use for training the model')
    number = st.number_input(
        label='Enter size of train set (X%)', min_value=0, max_value=100, value=30, step=1)
    number = 100 - number
    X_train, X_test, y_train, y_test = [], [], [], []
    # Compute the percentage of test and training data
    if (feature_predict_select in df.columns and oversample_select == 'Yes'):
        X_train, X_test, y_train, y_test = split_dataset(
            df, number, feature_predict_select, feature_input_select, oversample=True)
    if (feature_predict_select in df.columns and oversample_select == 'No'):
        X_train, X_test, y_train, y_test = split_dataset(
            df, number, feature_predict_select, feature_input_select, oversample=False)
    st.write('Number of entries in training set: {}'.format(X_train.shape[0]))
    st.write('Number of entries in testing set: {}'.format(X_test.shape[0]))

    classification_methods_options = ['Logistic Regression',
                                      'Logistic Regression (Newton Cholesky)',
                                      'K Nearest Neighbor',
                                      'Decision Tree',
                                      'Random Forest',
                                      'Naive Bayes',
                                      'Linear Support Vector Machines']

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
                max_value=100000,
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
        lg_model.fit(X_train, y_train)
        st.write('Logistic Regression Model trained')
        y_pred = lg_model.predict(X_test)
        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification','Area under the ROC curve'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        
        Metric_data = compute_evaluation(y_pred, y_test)
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

if (classification_methods_options[1] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[1])
        lg_model = LogisticRegression(penalty = 'l2',max_iter = 100000, class_weight='balanced', solver = 'newton-cholesky')
        lg_model.fit(X_train, y_train)
        y_pred = lg_model.predict(X_test)

        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification','Area under the ROC curve'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        Metric_data = compute_evaluation(y_pred, y_test)
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

if (classification_methods_options[2] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[2])
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
        Metric_data = compute_evaluation(y_pred, y_test)
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
        

if (classification_methods_options[3] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[3])
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
        Metric_data = compute_evaluation(y_pred, y_test)
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

if (classification_methods_options[4] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[4])
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
        Metric_data = compute_evaluation(y_pred, y_test)
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

if (classification_methods_options[5] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[5])
        lg_model = GaussianNB()
        lg_model.fit(X_train, y_train)
        st.write('Gaussian Naive Bayes Model trained')
        y_pred = lg_model.predict(X_test)
        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        Metric_data = compute_evaluation(y_pred, y_test)
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

if (classification_methods_options[6] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[6])
        lg_model = make_pipeline(StandardScaler(),LinearSVC(dual=False, random_state=0, tol=1e-5))
        lg_model.fit(X_train, y_train)
        st.write('Linear Support Vector Machine Model trained')
        y_pred = lg_model.predict(X_test)
        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity','Misclassification'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        
        Metric_data = compute_evaluation(y_pred, y_test)
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
