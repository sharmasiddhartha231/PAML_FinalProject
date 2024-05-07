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
    X, y = df.loc[:, df.columns.isin(input_var)], df.loc[:, df.columns.isin([target])]
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
    # Add code here
    metric_dict = {'Precision': -1,
                   'Sensitivity': -1,
                   'Accuracy': -1,
                   'Specificity': -1,
                   'Misclassification':-1}
    cmatrix = confusion_matrix(true_labels, prediction_labels, labels=[0,1])
    tn, fp, fn, tp = cmatrix.ravel()
    specificity = tn / (tn+fp)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    misclassification = (fp + fn)/(tp + tn + fp + fn)
    sensitivity = tp /(tp + fn)
    precision = tp / (tp + fp)
    metric_dict['Precision'] = precision
    metric_dict['Sensitivity'] = sensitivity
    metric_dict['Accuracy'] = accuracy
    metric_dict['Specificity'] = specificity
    metric_dict['Misclassification'] = misclassification
    return metric_dict

class LogisticRegression(object):
    def __init__(self, learning_rate=0.001, num_iterations=500): 
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.likelihood_history=[]
    
    # Checkpoint 5
    def predict_probability(self, X):
        '''
        Produces probabilistic estimate for P(y_i = +1 | x_i, w)
            Estimate ranges between 0 and 1.
        Input:
            - X: Input features
            - W: weights/coefficients of logistic regression model
            - b: bias or y-intercept of logistic regression classifier
        Output:
            - y_pred: probability of positive product review
        '''
        y_pred=None
        # Take dot product of feature_matrix and coefficients  
        # Add code here
        #num_features, num_examples = X.shape
        score = np.dot(X, self.W) +self.b
        # Add code here   
        y_pred = 1. / (1.+np.exp(-score))    
        return y_pred
    
    # Checkpoint 6
    def compute_avg_log_likelihood(self, X, Y, W):
        '''
        Compute the average log-likelihood of logistic regression coefficients

        Input
            - X: subset of features in dataset
            - Y: true sentiment of inputs
            - W: logistic regression weights
        Output
            - lp: log likelihood estimation
        '''
        lp=None
        # Add code here
        indicator = (Y==+1)
        scores = np.dot(X, W)
        logexp = np.log(1. + np.exp(-scores))
        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask]
        lp = np.sum((indicator-1)*scores - logexp)/len(X)
        return lp
    
    # Checkpoint 7
    def update_weights(self):      
        '''
        Compute the logistic regression derivative using 
        gradient ascent and update weights self.W

        Inputs: None
        Output: None
        '''
        num_examples, num_features = self.X.shape
        y_pred = self.predict(self.X)
        dW = self.X.T.dot(self.Y-y_pred) / num_examples 
        db = np.sum(self.Y-y_pred) / num_examples 
        self.b = self.b + self.learning_rate * db
        self.W = self.W + self.learning_rate * dW
        #for i in range(len(self.W)):
        #    y_pred = 1 / (1 + np.exp(-(self.X[:,i].dot(self.W[i]) + self.b))) 
        log_likelihood = self.compute_avg_log_likelihood(self.X, self.Y, self.W)
        self.likelihood_history.append(log_likelihood)
        return self
    
    # Checkpoint 8
    def predict(self, X):
        '''
        Hypothetical function  h(x)
        Input: 
            - X: Input features
            - W: weights/coefficients of logistic regression model
            - b: bias or y-intercept of logistic regression classifier
        Output:
            - Y: list of predicted classes 
        '''
        y_pred=0
        # Add code here
        Z = 1 / (1 + np.exp(- (self.X.dot(self.W) + self.b)))
        y_pred = [-1 if z <= 0.5 else +1 for z in Z]
        return y_pred 
    
    # Checkpoint 9
    def fit(self, X, Y):   
        '''
        Run gradient ascent to fit features to data using logistic regression 
        Input: 
            - X: Input features
            - Y: list of actual product sentiment classes 
            - num_iterations: # of iterations to update weights using gradient ascent
            - learning_rate: learning rate
        Output: None
        '''
        # Add code here
        self.X = X
        self.Y = Y
        num_examples, num_features = self.X.shape    
        W = np.zeros(num_features)
        self.W = W
        b = 0
        self.b = b
        for _ in range(self.num_iterations):          
            self.update_weights()   
        return self

    # Checkpoint 10
    def get_weights(self, model_name):
        '''
        This function prints the coefficients of the trained models
        
        Input:
            - model_name (list of strings): list of model names including: 'Logistic Regression', 'Stochastic Gradient Ascent with Logistic Regression' 
        Output:
            - out_dict: a dicionary contains the coefficients of the selected models, with the following keys:
            - 'Logistic Regression'
            - 'Stochastic Gradient Ascent with Logistic Regression'
        '''
        out_dict = {'Logistic Regression': [],
                    'Stochastic Gradient Ascent with Logistic Regression': []}
        
        # Add code here
        weights = self.fit(self.X, self.Y)
        if model_name == 'Logistic Regression':
            out_dict['Logistic Regression'] = weights
        if model_name == 'Stochastic Gradient Ascent with Logistic Regression':
            out_dict['Stochastic Gradient Ascent with Logistic Regression'] = weights
        return out_dict

#class StochasticLogisticRegression(LogisticRegression):
    #def __init__(self, num_iterations, learning_rate, batch_size): 
        self.likelihood_history=[]
        self.batch_size=batch_size

        # invoking the __init__ of the parent class
        LogisticRegression.__init__(self, learning_rate, num_iterations)

    # Checkpoint 11
    #def fit(self, X, Y):
        '''
        Run mini-batch stochastic gradient ascent to fit features to data using logistic regression 

        Input
            - X: input features
            - Y: target variable (product sentiment)
        Output: None
        '''
        # Add code here
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
        # Learning rate schedule
            self.learning_rate=self.learning_rate/1.02
        return self

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
                value='0.0001',
                key='lg_learning_rate_textinput'
            )
            st.write('You select the following learning rate value(s): {}'.format(lg_learning_rate_input))

        with (lg_col2):
            # Maximum iterations to run the LG until convergence
            lg_num_iterations = st.number_input(
                label='Enter the number of maximum iterations on training data',
                min_value=1,
                max_value=5000000,
                value=500,
                step=100,
                key='lg_max_iter_numberinput'
            )
            st.write('You set the maximum iterations to: {}'.format(lg_num_iterations))

        lg_params = {
            'num_iterations': lg_num_iterations,
            'learning_rate': [float(val) for val in lg_learning_rate_input.split(',')],
        }
        if st.button('Logistic Regression Model'):
            try:
                lg_model = LogisticRegression(num_iterations=lg_params['num_iterations'], 
                                            learning_rate=lg_params['learning_rate'][0])
                lg_model.fit(X_train.to_numpy(), np.ravel(y_train))
                st.session_state[classification_methods_options[0]] = lg_model
            except ValueError as err:
                st.write({str(err)})
        
        if classification_methods_options[0] not in st.session_state:
            st.write('Logistic Regression Model is untrained')
        else:
            st.write('Logistic Regression Model trained')

        st.markdown('### Evaluate your model')
        evaluation_options = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'Misclassification'] 
        evaluation_metric_select = st.multiselect(
        label='Select evaluation metric for current model',
        options=evaluation_options,
        key='evaluation_select'
        )
        st.session_state['evaluation'] = evaluation_metric_select
        if evaluation_metric_select in evaluation_options:
            Metric_data = compute_evaluation(y_pred, y_test, evaluation_metric_select)

if (classification_methods_options[1] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[1])
        ml_model = LogisticRegression(penalty='l2', max_iter = 100000, solver = 'newton-cholesky')
        ml_model.fit(X_train, y_train)
        y_pred = ml_model.predict(X_test)
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

if (classification_methods_options[2] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[2])

        ml_model = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
        ml_model.fit(X_train, y_train)
        y_pred = ml_model.predict(X_test)
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

if (classification_methods_options[3] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[3])

        ml_model = DecisionTreeClassifier()
        ml_model.fit(X_train, y_train)
        y_pred = ml_model.predict(X_test)
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

if (classification_methods_options[4] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[4])

        ml_model = RandomForestClassifier(random_state=0)
        ml_model.fit(X_train, y_train)
        y_pred = ml_model.predict(X_test)
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

if (classification_methods_options[5] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[5])

        ml_model = GaussianNB()
        ml_model.fit(X_train, y_train)
        y_pred = ml_model.predict(X_test)
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

if (classification_methods_options[6] == classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('## ' + classification_methods_options[6])

        ml_model = make_pipeline(StandardScaler(),LinearSVC(dual=False, random_state=0, tol=1e-5))
        ml_model.fit(X_train, y_train)
        y_pred = ml_model.predict(X_test)
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
