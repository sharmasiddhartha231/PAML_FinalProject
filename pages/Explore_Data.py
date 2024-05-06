import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

#############################################

st.markdown('# Explore Dataset')

#############################################

st.markdown(""" We are using a dataset from CDC called the Behavioral Risk Factor Surveillance System (BRFSS) data. This consists of randomized phone call based survey data from adults across the US and details various health and disease based metrics. We selected for specific metrics which have been known to be associated with Diabetes. The following Factors were selected.

- SEXVAR - Sex of respondent.
- GENHLTH - General Health of respondent.
- PRIMINSR - Primary source of your health care coverage?
- CHECKUP1 - Last visit to a doctor for routine checkup.
- EXERANY2 - Any physical exercise outside of work in last 30 days.
- BPHIGH6 - Ever told you had High Blood pressure.
- CVDINFR4 - Ever told you that you had a heart attack also called a myocardial infarction?
- CVDCRHD4 - Ever told you had angina or coronary heart disease?
- CVDSTRK3 - Ever told you had a stroke?
- CHCOCNCR - Ever told you had cancer?
- CHCCOPD3 - Ever told you had C.O.P.D. (chronic obstructive pulmonary disease), emphysema or chronic bronchitis?
- ADDEPEV3 - Ever told you had a depressive disorder (including depression, major depression, dysthymia, or minor depression)?
- CHCKDNY2 - Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?
- HAVARTH5 - Ever told you that you had some form of arthritis, rheumatoid arthritis, gout, lupus, or fibromyalgia?
- MARITAL - Marital Status.
- RENTHOM1 - Own or rent your home.
- EMPLOY1 - Employment Status.
- INCOME3 - Annual Household income.
- BLIND - Are you blind or do you have serious difficulty seeing, even when wearing glasses?
- DECIDE - Because of a physical, mental, or emotional condition, do you have serious difficulty concentrating, remembering, or making decisions?
- X_AGEG5YR - Age Categories.
- X_RACE - Race/ethnicity categories.
- X_EDUCAG - Education Level.
- X_SMOKE - Smoking Status.
- X_ASTHMS1 - Asthma Status.
- X_BMI5CAT - BMI Index.
- PHYSHLTH14D - Calculated variable for 3 levels for not having good physical health status.
- MENTHLTH14D - Calculated variable for 3 levels for not having good mental health status.
- ALCOFREQ - Alcohol Drink Frequency per 30 days.
- VACCSTAT - Vaccination Status.
- CHOLSTAT - Cholesterol Status.
- X_FRUTSU1DF - Number of fruit and fruit products consumed per day.
- X_VEGSU1DF - Number of green vegetables and other vegetables (excluding potatoes) consumed per day.
- DIABETERES - Whether you have Diabetes or not (Dependent Variable).
""")


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
    st.session_state['diabetes'] = data
    return data

# Helper Function
def compute_correlation(df, features):
    """
    This function computes pair-wise correlation coefficents of X and render summary strings

    Input: 
        - df: pandas dataframe 
        - features: a list of feature name (string), e.g. ['age','height']
    Output: 
        - correlation: correlation coefficients between one or more features
        - summary statements: a list of summary strings where each of it is in the format: 
            '- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}'
    """
    correlation = df[features].corr()
    feature_pairs = combinations(features, 2)
    cor_summary_statements = []

    for f1, f2 in feature_pairs:
        cor = correlation[f1][f2]
        summary = '- Features %s and %s are %s %s correlated: %.2f' % (
            f1, f2, 'strongly' if cor > 0.5 else 'weakly', 'positively' if cor > 0 else 'negatively', cor)
        st.write(summary)
        cor_summary_statements.append(summary)

    return correlation, cor_summary_statements

###################### FETCH DATASET #######################
df = None
if('diabetes' in st.session_state):
    df = st.session_state['diabetes']
else:
    filepath = "/Users/siddharthasharma/Desktop/PAML/PAML_FinalProject/Diabetes_Data_Sub_Strict_Main_String.txt"
    if(filepath):
        df = load_dataset(filepath)

######################### MAIN BODY #########################

######################### EXPLORE DATASET #########################

if df is not None:
    st.markdown('### 1. Explore Dataset Features')

    # Restore dataset if already in memory
    st.session_state['diabetes'] = df

    # Display dataframe as table
    st.dataframe(df)

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 2. Visualize Features')

    numeric_columns = list(df.columns)
    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label='Type of chart',
        options=['Histogram', 'Pie Chart', 'Stacked Bar Chart', 'Percent Bar Chart']
    )
    st.sidebar.header('Select dataset')
    if chart_select == 'Histogram' or chart_select == 'Pie Chart':
        data_select = st.sidebar.selectbox(
            label='Type of chart',
            options=['Full Dataset', 'Diabetes Only', 'PreDiabetes Only', 'Non Diabetes']
        )
    if chart_select == 'Stacked Bar Chart' or chart_select == 'Percent Bar Chart':
        data_select = st.sidebar.selectbox(
            label='Type of chart',
            options=['Full Dataset']
        )

    st.sidebar.header('Specify Input Variable')
    # Draw plots
    if chart_select == 'Histogram' and data_select == 'Full Dataset':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.histogram(data_frame=df,
                                x=x_values,color_discrete_sequence=['indianred'])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Histogram' and data_select == 'Diabetes Only':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.histogram(data_frame=df[df["DIABETERES"] == "Diabetes"],
                                x=x_values,color_discrete_sequence=['indianred'])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Histogram' and data_select == 'PreDiabetes Only':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.histogram(data_frame=df[df["DIABETERES"] == "Prediabetes"],
                                x=x_values,color_discrete_sequence=['indianred'])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Histogram' and data_select == 'Non Diabetes':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.histogram(data_frame=df[df["DIABETERES"] == "No Diabetes"],
                                x=x_values,color_discrete_sequence=['indianred'])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Pie Chart' and data_select == 'Full Dataset':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            d1 = df[x_values].value_counts().reset_index()
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.pie(data_frame=d1,
                                values='count',names = x_values,color_discrete_sequence=px.colors.sequential.RdBu)
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Pie Chart' and data_select == 'Diabetes Only':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            d1 = df[df["DIABETERES"] == "Diabetes"]
            d1 = d1[x_values].value_counts().reset_index()
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.pie(data_frame=d1,
                                values='count',names = x_values,color_discrete_sequence=px.colors.sequential.RdBu)
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Pie Chart' and data_select == 'PreDiabetes Only':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            d1 = df[df["DIABETERES"] == "Prediabetes"]
            d1 = d1[x_values].value_counts().reset_index()
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.pie(data_frame=d1,
                                values='count',names = x_values,color_discrete_sequence=px.colors.sequential.RdBu)
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Pie Chart' and data_select == 'Non Diabetes':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            d1 = df[df["DIABETERES"] == "No Diabetes"]
            d1 = d1[x_values].value_counts().reset_index()
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.pie(data_frame=d1,
                                values='count',names = x_values,color_discrete_sequence=px.colors.sequential.RdBu)
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Stacked Bar Chart' and data_select == 'Full Dataset':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            d1 = df.value_counts([x_values, 'DIABETERES']).reset_index()
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.bar(data_frame=d1,x='count',y=x_values,color = 'DIABETERES')
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Percent Bar Chart' and data_select == 'Full Dataset':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            d1 = df.value_counts([x_values, 'DIABETERES']).reset_index()
            d1['pct'] = 100 * d1['count']/d1.groupby(x_values)['count'].transform('sum')
            #side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.bar(data_frame=d1,x='pct',y=x_values,color = 'DIABETERES')
            st.write(plot)
        except Exception as e:
            print(e)
    
###################### CORRELATION ANALYSIS #######################
    st.markdown("### 11. Correlation Analysis")
    # Collect features for correlation analysis using multiselect
    numeric_columns = list(df.select_dtypes(['float','int']).columns)


    select_features_for_correlation = st.multiselect(
        'Select features for visualizing the correlation analysis (up to 4 recommended)',
        numeric_columns,
    )

    # Compute correlation between selected features
    correlation, correlation_summary = compute_correlation(
        df, select_features_for_correlation)
    st.write(correlation)

    # Display correlation of all feature pairs
    if select_features_for_correlation:
        try:
            fig = scatter_matrix(
                df[select_features_for_correlation], figsize=(12, 8))
            st.pyplot(fig[0][0].get_figure())
        except Exception as e:
            print(e)

    st.markdown('#### Continue to Preprocess Data')