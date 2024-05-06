import numpy as np                    
from sklearn.model_selection import train_test_split
import streamlit as st                  
import random
from helper_functions import fetch_dataset, set_pos_neg_reviews
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# set seed=10 to produce consistent results
random.seed(10)

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown(
    "### Homework 3 - Predicting Product Review Sentiment Using Classification")

#############################################

st.title('Train Model')

#############################################

# Checkpoint 4
def split_dataset(df, number, target, feature_encoding, random_state=42):
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
    X_val = []
    y_train = []
    y_val = []
    X_train_sentiment, X_val_sentiment = [], []
    
    # Add code here
    X, y = df.loc[:, ~df.columns.isin([target])], df.loc[:, df.columns.isin([target])]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=number/100, random_state=random_state)
    if('Word Count' in feature_encoding):
        X_train_sentiment = X_train.loc[:, X_train.columns.str.startswith('word_count_')]
        X_val_sentiment = X_val.loc[:, X_val.columns.str.startswith('word_count_')]
    if('TF-IDF' in feature_encoding):
        X_train_sentiment = X_train.loc[:, X_train.columns.str.startswith('tf_idf_word_count_')]
        X_val_sentiment = X_val.loc[:, X_val.columns.str.startswith('tf_idf_word_count_')]
    return X_train_sentiment, X_val_sentiment, y_train, y_val

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

class StochasticLogisticRegression(LogisticRegression):
    def __init__(self, num_iterations, learning_rate, batch_size): 
        self.likelihood_history=[]
        self.batch_size=batch_size

        # invoking the __init__ of the parent class
        LogisticRegression.__init__(self, learning_rate, num_iterations)

    # Checkpoint 11
    def fit(self, X, Y):
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
    st.dataframe(df)

    # Select positive and negative ratings
    pos_neg_select = st.slider(
        'Select a range of ratings for negative reviews',
        1, 5, 3,
        key='pos_neg_selectbox')

    if (pos_neg_select and st.button('Set negative sentiment upper bound')):
        df = set_pos_neg_reviews(df, pos_neg_select)

        st.write('You selected ratings positive rating greater than {}'.format(
            pos_neg_select))

    # Select variable to predict
    feature_predict_select = st.selectbox(
        label='Select variable to predict',
        index=df.columns.get_loc(
            'sentiment') if 'sentiment' in df.columns else 0,
        options=df.columns,
        key='feature_selectbox',
    )

    st.session_state['target'] = feature_predict_select

    word_count_encoder_options=[]
    word_count_data = df.loc[:, df.columns.str.startswith('word_count_')]
    if(len(word_count_data)):
        word_count_encoder_options.append('Word Count')

    tfidf_word_count_data = df.loc[:, df.columns.str.startswith('tfidf_word_count_')]
    if(len(tfidf_word_count_data)):
        word_count_encoder_options.append('TF-IDF')
    
    if ('word_encoder' in st.session_state):
        if (st.session_state['word_encoder'] is not None):
            st.write('Restoring selected encoded features {}'.format(
                word_count_encoder_options))

    # Select input features
    feature_input_select = st.selectbox(
        label='Select word encoder for classification input',
        options=word_count_encoder_options,
        key='feature_select'
    )

    st.session_state['feature'] = feature_input_select

    st.write('You selected input {} and output {}'.format(
        feature_input_select, feature_predict_select))

    # Task 4: Split train/test
    st.markdown('## Split dataset into Train/Test sets')
    st.markdown(
        '### Enter the percentage of test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    X_train, X_val, y_train, y_val = [], [], [], []
    # Compute the percentage of test and training data
    if (feature_predict_select in df.columns):
        X_train, X_val, y_train, y_val = split_dataset(
            df, number, feature_predict_select, feature_input_select)

    classification_methods_options = ['Logistic Regression',
                                      'Stochastic Gradient Ascent with Logistic Regression']

    trained_models = [
        model for model in classification_methods_options if model in st.session_state]

    st.session_state['trained_models'] = trained_models
    
    # Collect ML Models of interests
    classification_model_select = st.multiselect(
        label='Select regression model for prediction',
        options=classification_methods_options,
    )
    st.write('You selected the follow models: {}'.format(
        classification_model_select))

    # Add parameter options to each regression method

    # Task 5: Logistic Regression
    if (classification_methods_options[0] in classification_model_select):# or classification_methods_options[0] in trained_models):
        st.markdown('#### ' + classification_methods_options[0])

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
                max_value=5000,
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

    # Task 6: Stochastic Gradient Ascent with Logistic Regression
    if (classification_methods_options[1] in classification_model_select):
        st.markdown('#### ' + classification_methods_options[1])

        # Number of iterations: maximum iterations to run the iterative SGD
        sdg_num_iterations = st.number_input(
            label='Enter the number of maximum iterations on training data',
            min_value=1,
            max_value=5000,
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
        st.write('You select the following learning rate: {}'.format(sdg_learning_rate))

        # tolerance: stopping criteria for iterations
        sgd_batch_size = st.text_input(
            label='Input a batch size value',
            value='50',
            key='sgd_batch_size_textinput'
        )
        sgd_batch_size = int(sgd_batch_size)
        st.write('You select the following batch_size: {}'.format(sgd_batch_size))

        sgd_params = {
            'num_iterations': sdg_num_iterations,
            'batch_size': sgd_batch_size,
            'learning_rate': sdg_learning_rate,
        }

        if st.button('Train Stochastic Gradient Ascent Model'):
            try:
                sdg_model = StochasticLogisticRegression(num_iterations=sgd_params['num_iterations'], 
                                                        learning_rate=sgd_params['learning_rate'],
                                                        batch_size=sgd_params['batch_size'])
                sdg_model.fit(X_train.to_numpy(), np.ravel(y_train))
                st.session_state[classification_methods_options[1]] = sdg_model
            except ValueError as err:
                st.write({str(err)})
        if classification_methods_options[1] not in st.session_state:
            st.write('Stochastic Gradient Ascent Model is untrained')
        else:
            st.write('Stochastic Gradient Ascent Model trained')

    # Store models in dict
    trained_models={}
    for model_name in classification_methods_options:
        if(model_name in st.session_state):
            trained_models[model_name] = st.session_state[model_name]


    # Task 9: Inspect classification coefficients
    st.markdown('## Inspect model coefficients')

    # Select multiple models to inspect
    inspect_models = st.multiselect(
        label='Select features for classification input',
        options=classification_model_select,
        key='inspect_multiselect'
    )
    st.write('You selected the {} models'.format(inspect_models))

    models = {}
    weights_dict = {}
    if(inspect_models):
        for model_name in inspect_models:
            weights_dict = trained_models[model_name].get_weights(model_name)

    # Inspect model likelihood
    st.markdown('## Inspect model likelihood')

    # Select multiple models to inspect
    inspect_model_likelihood = st.selectbox(
        label='Select model',
        options=classification_model_select,
        key='inspect_cost_multiselect'
    )

    st.write('You selected the {} model'.format(inspect_model_likelihood))

    if(inspect_model_likelihood):
        try:
            fig = make_subplots(rows=1, cols=1,
                shared_xaxes=True, vertical_spacing=0.1)
            cost_history=trained_models[inspect_model_likelihood].likelihood_history

            x_range = st.slider("Select x range:",
                                    value=(0, len(cost_history)))
            st.write("You selected : %d - %d"%(x_range[0],x_range[1]))
            cost_history_tmp = cost_history[x_range[0]:x_range[1]]
            
            fig.add_trace(go.Line(x=np.arange(x_range[0],x_range[1],1),
                        y=cost_history_tmp, mode='lines', name=inspect_model_likelihood), row=1, col=1)

            fig.update_xaxes(title_text="Training Iterations")
            fig.update_yaxes(title_text='Log Likelihood', row=1, col=1)
            fig.update_layout(title=inspect_model_likelihood)
            st.plotly_chart(fig)
        except Exception as e:
            print(e)

    st.write('Continue to Test Model')
