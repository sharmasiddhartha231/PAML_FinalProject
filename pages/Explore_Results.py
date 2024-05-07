import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

#############################################

st.markdown('# Explore Results')

#############################################

st.markdown('On this page, we present the results from the various machine learning models we tested to check how they perform when predicting using the entire set of variables. We also present the results of for the association of these different variables with increased risk of diabetes. The results for the association of different factors to diabetes risk is presented as Odds Ratio calculated using a Univariable Logistic Regression model.')
st.markdown('Please do keep in mind that the association of different factors to diabetes risk is simply correlative and does not imply causation.')

st.markdown('If you wish to train the various machine learning models using different sets of parameters by yourself, please switch to the **Model Exploration** page. More details regarding these methods are provided there for your assistance.')


# Helper Function
def load_dataset(filepath):
    """
    This function uses the filepath (string) a .csv file locally on a computer 
    to import a dataset with pandas read_csv() function. Then, store the 
    dataset in session_state.

    Input: data is the filename or path to file (string)
    Output: pandas dataframe df
    """
    data = pd.read_csv(filepath)
    st.session_state['data'] = data
    return data


###################### FETCH DATASET #######################
df = None
if('data' in st.session_state):
    df = st.session_state['data']
else:
    filepath = "/Users/siddharthasharma/Desktop/PAML/PAML_FinalProject/Diabetes_Data_Sub_Strict_Main_String_New.txt"
    if(filepath):
        df = load_dataset(filepath)

######################### MAIN BODY #########################

######################### EXPLORE DATASET #########################

if df is not None:
    # Restore dataset if already in memory
    st.session_state['data'] = df

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 2. Visualize Results')
