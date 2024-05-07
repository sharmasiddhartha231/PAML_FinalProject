import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

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
        X_SMOKER = st.selectbox(
            label='What is your smoking status?',
            options=['Current Smoker (Daily)', 'Current Smoker (Some days)','Former Smoker', 'Non Smoker'],
            key='X_SMOKER'
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
            options=['Have asthma currently', 'Had asthma before', 'Never hae had asthma'],
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

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 2. Choose a model') 
    classification_methods_options = ['Logistic Regression',
                                      'Logistic Regression (Newton Cholesky)',
                                      'K Nearest Neighbor',
                                      'Decision Tree',
                                      'Random Forest',
                                      'Naive Bayes',
                                      'Linear Support Vector Machines']
    classification_model_select = st.selectbox(
        label='Select classification model for prediction',
        options=classification_methods_options,
    )

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 3. Check your results') 

        
        
        
        
