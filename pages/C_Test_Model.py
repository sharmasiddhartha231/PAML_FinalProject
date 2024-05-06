import numpy as np                   
import pandas as pd               
import streamlit as st                 
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helper_functions import fetch_dataset, apply_threshold
from pages.B_Train_Model import split_dataset
from sklearn.metrics import confusion_matrix
random.seed(10)
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown(
    "### Homework 3 - Predicting Product Review Sentiment Using Classification")

#############################################

st.title('Test Model')

#############################################

# Checkpoint 12
def compute_accuracy(prediction_labels, true_labels):    
    '''
    Compute classification accuracy
    Input
        - prediction_labels (numpy): predicted product sentiment
        - true_labels (numpy): true product sentiment
    Output
        - accuracy (float): accuracy percentage (0-100%)
    '''
    accuracy=0
    num_correct = np.sum(prediction_labels == true_labels)
    accuracy = num_correct / len(true_labels)
    # Add code here
    return accuracy

# Checkpoint 13
def compute_precison_recall(prediction_labels, true_labels, print_summary=False):
    '''
    Compute precision and recall 
    Input
        - prediction_labels (numpy): predicted product sentiment
        - true_labels (numpy): true product sentiment
    Output
        - precision (float): precision score = TP/TP+FP
        - recall (float): recall score = TP/TP+FN
    '''
    precision=0
    recall=0
    # Add code here
    cmatrix = confusion_matrix(true_labels, prediction_labels)
    true_neg, false_pos, false_neg, true_pos = cmatrix.ravel()
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos / (true_pos + false_neg)
    return precision, recall

# Helper Function
def compute_eval_metrics(X, y_true, model, metrics, print_summary=False):
    """
    This function computes one or more metrics (precision, recall, accuracy) using the model

    Input:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evaluate performance (string); 'precision', 'recall', 'accuracy'
    Output:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {'precision': -1,
                   'recall': -1,
                   'accuracy': -1}

    # Predict the product sentiment using the input model and data X
    y_pred = model.predict(X)

    # Compute the evaluation metrics in 'metrics = ['precision', 'recall', 'accuracy']' using the predicted sentiment
    precision, recall = compute_precison_recall(y_pred, y_true.to_numpy())
    accuracy = compute_accuracy(y_pred, y_true.to_numpy())

    if 'precision' in metrics:
        metric_dict['precision'] = precision
    if 'recall' in metrics:
        metric_dict['recall'] = recall
    if 'accuracy' in metrics:
        metric_dict['accuracy'] = accuracy

    return metric_dict

# Helper Function
def plot_roc_curve(X_train, X_val, y_train, y_val, trained_models, model_names):
    """
    Plot the ROC curve between predicted and actual values for model names in trained_models on the training and validation datasets

    Input:
        - X_train: training input data
        - X_val: test input data
        - y_true: true targets
        - y_pred: predicted targets
        - trained_model_names: trained model names
        - trained_models: trained models in a dictionary (accessed with model name)
    Output:
        - fig: the plotted figure
        - df: a dataframe containing the train and validation errors, with the following keys:
            - df[model_name.__name__ + " Train Precision"] = train_precision_all
            - df[model_name.__name__ + " Train Recall"] = train_recall_all
            - df[model_name.__name__ + " Validation Precision"] = val_precision_all
            - df[model_name.__name__ + " Validation Recall"] = val_recall_all
    """
    # Set up figures
    fig = make_subplots(rows=len(trained_models), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)

    # Intialize variables
    df = pd.DataFrame()
    threshold_values = np.linspace(0.5, 1, num=100)

    for i, trained_model in enumerate(trained_models):
        model_name = model_names[i]

        train_precision_all = []
        train_recall_all = []
        val_precision_all = []
        val_recall_all = []
        train_probabilities = []
        val_probabilities = []

        # Make predictions on the train and test set using predict_probability() function
        train_probabilities = trained_model.predict_probability(X_train)
        val_probabilities = trained_model.predict_probability(X_val)

        # Computer precision and recall on training set using threshold_values
        # Computer precision and recall on validation set using threshold_values
        for threshold in threshold_values:
            train_predictions = apply_threshold(train_probabilities, threshold)
            val_predictions = apply_threshold(val_probabilities, threshold)
            train_predictions = train_predictions.reshape(-1,1)
            val_predictions = val_predictions.reshape(-1,1)
            
            train_precision, train_recall = compute_precison_recall(train_predictions, y_train.to_numpy())
            train_precision = np.array(train_precision).reshape(-1,1)
            train_recall = train_recall.reshape(-1,1)

            train_precision_all.append(train_precision)
            train_recall_all.append(train_recall)

            val_precision, val_recall = compute_precison_recall(val_predictions, y_val.to_numpy())
            val_precision = np.array(val_precision).reshape(-1,1)
            val_recall = val_recall.reshape(-1,1)
            
            val_precision_all.append(val_precision)
            val_recall_all.append(val_recall)

        train_precision_all = np.ravel(train_precision_all)
        train_recall_all = np.ravel(train_recall_all)
        val_precision_all = np.ravel(val_precision_all)
        val_recall_all = np.ravel(val_recall_all)
            
        # Plot ROC Curve
        fig.add_trace(go.Scatter(x=train_recall_all,
                                 y=train_precision_all, name=model_names[i]+" Train"), row=i+1, col=1)

        fig.add_trace(go.Scatter(x=val_recall_all,
                                 y=val_precision_all, name=model_names[i]+" Validation"), row=i+1, col=1)

        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text='Precision', row=i+1, col=1)
        fig.update_layout(title=model_name+' ROC Curve')

        # Save output values
        df[model_name+" Train Precision"] = train_precision_all
        df[model_name+" Train Recall"] = train_recall_all
        df[model_name+" Validation Precision"] = val_precision_all
        df[model_name+" Validation Recall"] = val_recall_all
    return fig, df

# Helper function
def restore_data_splits(df):
    """
    This function restores the training and validation/test datasets from the training page using st.session_state
                Note: if the datasets do not exist, re-split using the input df

    Input: 
        - df: the pandas dataframe
    Output: 
        - X_train: the training features
        - X_val: the validation/test features
        - y_train: the training targets
        - y_val: the validation/test targets
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    # Restore train/test dataset
    if ('X_train' in st.session_state):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.write('Restored train data ...')
    if ('X_val' in st.session_state):
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        st.write('Restored test data ...')
    if (X_train is None):
        # Select variable to explore
        numeric_columns = list(df.select_dtypes(include='number').columns)
        feature_select = st.selectbox(
            label='Select variable to predict',
            options=numeric_columns,
        )
        X = df.loc[:, ~df.columns.isin([feature_select])]
        Y = df.loc[:, df.columns.isin([feature_select])]

        # Split train/test
        st.markdown(
            '### Enter the percentage of test data to use for training the model')
        number = st.number_input(
            label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

        X_train, X_val, y_train, y_val = split_dataset(X, Y, number, feature_select, 'TF-IDF')
        st.write('Restored training and test data ...')
    return X_train, X_val, y_train, y_val

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:
    # Restore dataset splits
    X_train, X_val, y_train, y_val = restore_data_splits(df)

    st.markdown("## Get Performance Metrics")
    metric_options = ['precision', 'recall', 'accuracy']

    classification_methods_options = ['Logistic Regression',
                                      'Stochastic Gradient Ascent with Logistic Regression']

    trained_models = [
        model for model in classification_methods_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained classification models for evaluation',
        options=trained_models
    )
    if (model_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button('Evaluate your selected classification models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('## Review Classification Model Performance')

            plot_options = ['ROC Curve', 'Metric Results']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=plot_options
            )
            ############## Task 10: Compute evaluation metrics
            if 'ROC Curve' in review_plot:
                trained_select = [st.session_state[model]
                                  for model in model_select]
                fig, df = plot_roc_curve(
                    X_train, X_val, y_train, y_val, trained_select, model_select)
                st.plotly_chart(fig)

            ############## Task 11: Plot ROC Curves
            if 'Metric Results' in review_plot:
                models = [st.session_state[model]
                          for model in model_select]

                train_result_dict = {}
                val_result_dict = {}

                # Select multiple metrics for evaluation
                metric_select = st.multiselect(
                    label='Select metrics for classification model evaluation',
                    options=metric_options,
                )
                if (metric_select):
                    st.session_state['metric_select'] = metric_select
                    st.write(
                        'You selected the following metrics: {}'.format(metric_select))

                    for idx, model in enumerate(models):
                        train_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_train, y_train, model, metric_select, print_summary=True)
                        val_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_val, y_val, model, metric_select, print_summary=True)

                    st.markdown('### Predictions on the training dataset')
                    st.dataframe(train_result_dict)

                    st.markdown('### Predictions on the validation dataset')
                    st.dataframe(val_result_dict)