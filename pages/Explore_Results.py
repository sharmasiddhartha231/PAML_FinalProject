import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#############################################

st.markdown('# Explore Results')

#############################################

st.markdown('On this page, we present the results from the various machine learning models we tested to check how they perform when predicting using the entire set of variables. We also present the results of for the association of these different variables with increased risk of diabetes. The results for the association of different factors to diabetes risk is presented as Odds Ratio calculated using a Univariable Logistic Regression model.')
st.markdown('Please do keep in mind that the association of different factors to diabetes risk is simply correlative and does not imply causation.')
st.markdown('Another thing to keep in mind is that the association of different risk factors was conducted independently. For example, when associating age with diabetes risk, we excluded every other factor to calculate the odds of getting diabetes with age.')

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

def OddsCalculation(df, input_var,random_state=42):
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
    # Add code here
    df = df.drop(df[df.DIABETERES == 'Prediabetes'].index)
    df.DIABETERES[df.DIABETERES == 'No Diabetes'] = 0
    df.DIABETERES[df.DIABETERES == 'Diabetes'] = 1
    df = df.reset_index(drop=True) 
    df = df[['DIABETERES',input_var]]
    cols = [input_var]
    for i in cols:
        i = pd.get_dummies(df[i], drop_first=False)
        df = pd.concat([df,i], axis=1)
    x_data = df.drop(['DIABETERES'],axis=1)
    x_data = x_data.drop([input_var],axis=1)
    y_data = df['DIABETERES']
    y_data=y_data.astype('int')
    x_data = x_data.replace(False,0, regex=True)
    x_data = x_data.replace(True,1, regex=True)
    #X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=number/100, random_state=random_state)
    logmodel = LogisticRegression(max_iter = 100000, class_weight='balanced', solver = 'newton-cholesky', penalty='l2')
    logmodel.fit(x_data, y_data)
    Weights = pd.DataFrame({'Factor':x_data.columns, 'Odds Ratio':logmodel.coef_.flatten()})
    return Weights
    #return X_train, X_test, y_train, y_test

###################### FETCH DATASET #######################
df = None
if('data' in st.session_state):
    df = st.session_state['data']
else:
    filepath = "/Users/siddharthasharma/Desktop/PAML/PAML_FinalProject/Diabetes_Data_Sub_Strict_Main_String_New.txt"
    df = load_dataset(filepath)

######################### MAIN BODY #########################

######################### EXPLORE DATASET #########################

if df is not None:
    st.session_state['data'] = df

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 1. Our results, in a nutshell.')

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 2. Factors with the highest risk association.')

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 3. Association of each factor with diabetes prevalence.')
    st.markdown(""" To see the explanation for each factor, please refer to the **Explore Data** page
    """)
    colnames = df.drop(['DIABETERES'],axis=1).columns
    assc_select = st.selectbox(
        label='Select Variable to test association for.',
        options=colnames
    )
    if assc_select == 'SEXVAR':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Female'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'X_AGEG5YR':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == '18-24 YO'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'CHOLSTAT':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Low Cholesterol and do not take medicine'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'VACCSTAT':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Not vaccinated for either'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'GENHLTH':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Excellent Health'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'PRIMINSR':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'No Insurance'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'X_VEGSU1DF':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Less than once a day (Vegetable Consumption)'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'X_FRUTSU1DF':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Less than once a day (Fruit Consumption)'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'ALCOFREQ':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Dont drink (0 days)'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'MENTHLTH14D':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == '0 Days of bad mental health'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'PHYSHLTH14D':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == '0 Days of bad physical health'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'X_BMI5CAT':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Normal weight'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'X_ASTHMS1':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Never have had asthma'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'X_SMOKE':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Non Smoker'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'CHCOCNCR':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have cancer'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'X_EDUCAG':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not graduate high school'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'X_RACE':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'White'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'DECIDE':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'No difficulty deciding'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'BLIND':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'No difficulty seeing'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'INCOME3':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Less than 10K'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'EMPLOY1':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Employed'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'RENTHOM1':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Own home'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'MARITAL':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Married'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'HAVARTH5':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have arthritis'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'CHCKDNY2':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have kidney disease'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'CHCCOPD3':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have COPD'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'ADDEPEV3':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have depressive order'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'CVDSTRK3':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have a stroke'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'CVDCRHD4':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have coronary heart disease'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'CVDINFR4':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have a heart attack'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'BPHIGH6':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have high BP'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'EXERANY2':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Exercised'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
    if assc_select == 'CHECKUP1':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Last checkup was less than a year ago'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)