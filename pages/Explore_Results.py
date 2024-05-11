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
        st.markdown("""Sex of the respondent.""")
        st.table(weights)
        st.markdown(""" We see that men have a slightly higher chance of having diabetes (~1.2 times) as compared to women.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'X_AGEG5YR':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == '18-24 YO'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Age group of the respondent.""")
        st.table(weights)
        st.markdown(""" We see that the chances of diabetes increase with age, with people in the 50-54 year age group being nearly 10 times likely to have diabetes as compared to people in the 18-24 year age group. People in the 75-79 year age group are nearly 20 times more likely to have diabetes as compared to the youngest group. This shows there is a marked correlation between age and diabetes.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'CHOLSTAT':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Low Cholesterol and do not take medicine'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Cholesterol Status of the respondent.""")
        st.table(weights)
        st.markdown(""" We expected to see an increase in diabetes case in patients with high cholesterol as patients with diabetes usually have high levels of cholesterol. While there is some correlation there, we see a marked increase in diabetes in respondents who take cholesterol medication. People who take cholesterol medication despite having low cholesterol had the highest chances of having diabetes (~8 times compared to people with low cholesterol who do not take medication). This points in the direction that statins (usually found in cholesterol drugs) have a considerably higher risk association with diabetes compared to high cholesterol levels itself.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'VACCSTAT':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Not vaccinated for either'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Vaccination status of the respondent.""")
        st.table(weights)
        st.markdown(""" We see people who have taken the pneumonia vaccine have a higher chance of having diabetes (~2.7 times). This increases to over 3 times when respondents have taken both the flu and pneumonia vaccine despite the flu vaccine on its own not having any major association with diabetes on its own. 
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'GENHLTH':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Excellent Health'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""General Health of the respondent.""")
        st.table(weights)
        st.markdown("""People who reported having having poor health in general had increased diabetes cases. While the correlation makes sense, the poor health could be due to a number of factors, including having diabetes itself. 
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'PRIMINSR':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'No Insurance'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Healthcare coverage of the respondent.""")
        st.table(weights)
        st.markdown(""" We see people having Medicare, Military and IHS insurance have the highest chances of having diabetes. Medicare is used for elderly people, who have a high risk factor of diabetes, while IHS is used to provide insurance for American Indians and Alaskan natives, who have also been known to be at higher risk for Diabetes compared to other races. People using Military insurance might also have other conditions contributing to increased risk of diabetes.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'X_VEGSU1DF':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Less than once a day (Vegetable Consumption)'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Vegetable consumption (excluding potatoes) of the respondent.""")
        st.table(weights)
        st.markdown(""" We see a decreased correlation as the number of vegetables consumed per day increases.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'X_FRUTSU1DF':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Less than once a day (Fruit Consumption)'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Fruit and fruit product consumption o the respondent.""")
        st.table(weights)
        st.markdown(""" We see a decreased correlation as the number of fruits and fruit products consumed per day increases.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)

    if assc_select == 'ALCOFREQ':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Dont drink (0 days)'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Frequency of alcohol consumption of the respondent.""")
        st.table(weights)
        st.markdown("""The trend presented here is unexpected. We see a decrease in the chances of getting diabetes with people who consume alcohol often. One of the possible explanations for this might be that people with diabetes do not consume alcohol since alcohol is known to raise blood sugar levels and thus there is a higher prevalence of non drinkers amongst diabetic people.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)

    if assc_select == 'MENTHLTH14D':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == '0 Days of bad mental health'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Number of bad mental health days reported by the respondent.""")
        st.table(weights)
        st.markdown(""" We do not see a marked increase in diabetes risk even when the respondents had a lot more bad mental health days as compared to respondents who did not have any.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'PHYSHLTH14D':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == '0 Days of bad physical health'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Number of bad physical health days reported by the respondent.""")
        st.table(weights)
        st.markdown(""" We see a marked increase in diabetes risk (~3 times) in respondents who averaged 14 or more days a month regarding bad physical health days. 
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'X_BMI5CAT':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Normal weight'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""BMI of the respondent.""")
        st.table(weights)
        st.markdown(""" Obese people are over 4 times more likely to have diabetes compared to people with a regular BMI. Obesity has been known to raise the risk for diabetes alongside various other conditions, including high blood pressure and high cholesterol, all known factors associated with diabetes risk.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'X_ASTHMS1':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Never have had asthma'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Asthma status of the respondent.""")
        st.table(weights)
        st.markdown(""" We see that people who have asthma currently are 1.5 times more likely to have diabetes compared to people who have never had asthma before.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'X_SMOKE':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Non Smoker'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Sex of the respondent.""")
        st.table(weights)
        st.markdown(""" Like the correlation with alcohol frequency, we see that former smokers are more likely to be diabetic compared to non smokers. This again might be due to prior smokers having quit due to being diabetic. 
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'CHCOCNCR':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have cancer'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Did the respondent have cancer.""")
        st.table(weights)
        st.markdown(""" We see that respondents who had cancer were slightly more likely to have diabetes (~1.6 times) as compared to respondents who did not.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'X_EDUCAG':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not graduate high school'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Education status the respondent.""")
        st.table(weights)
        st.markdown(""" We see that diabetes prevalence decreases as education levels increase, with this factor being more socio-economic in nature, as higher education levels might indicate higher incomes and better access to healthcare, which might contribute to lower prevalence of diabetes in these respondents.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)

    if assc_select == 'X_RACE':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'White'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Race of the respondent.""")
        st.table(weights)
        st.markdown(""" American Indians and Alaskan Natives had the highest possible risk of getting diabetes compared to a Caucasian control. Respondents who identified as Black were the 2nd highest at risk (~1.8 times) compared to the same control. On the flipside, Asians had the least risk of diabetes compared to every other race.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'DECIDE':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'No difficulty deciding'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Does the respondent have difficulty making decisions.""")
        st.table(weights)
        st.markdown(""" People who have difficulty making decisions have a nearly 1.8 times higher risk of having diabetes as compared to people who do not. Diabetes has been known to cause problems with memory and learning, concentration which in general affect our ability to make decisions. 
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'BLIND':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'No difficulty seeing'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Does the respondent have difficulty seeing (even despite glasses).""")
        st.table(weights)
        st.markdown(""" People who have difficulty seeing have a nearly 2.5 times higher risk of having diabetes as compared to people who do not. Diabetes is known to cause vision loss and even blindness in certain cases.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'INCOME3':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Less than 10K'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Income of the respondent.""")
        st.table(weights)
        st.markdown(""" We see a marked decrease (over 5 times) in diabetes risk in the highest income category compared to the lowest income category. The trend generally follows as people with higher income have lower likelihood of having diabetes compared to people with lower income. This might be due to access of healthcare is easier for people in the higher income brackets which provides them with better care.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'EMPLOY1':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Employed'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Employment status of the respondent.""")
        st.table(weights)
        st.markdown(""" People unable to work were over 5 times more likely of having diabetes compared to people who have a job. People unemployed for over a year had the 2nd highest risk association with diabetes (2 times more likely). Being unemployed or unable to work both might be associated with disabilities, illnesses or other conditions, thus being contributory to higher diabetes risk or even might be caused due to it. Lack of employment might also tie in directly with low incomes and thus cause difficulty with having access to healthcare.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'RENTHOM1':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Own home'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Home owner status of the respondent.""")
        st.table(weights)
        st.markdown(""" We do not see any significant trend amongst respondents who owned or rented their homes.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'MARITAL':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Married'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Marital status of the respondent.""")
        st.table(weights)
        st.markdown(""" We see an increase in diabetes risk (~1.75 times more likely) amongst widowed respondents compared to married people. This correlation might arise due to widowed respondents being possibly older in age compared to the married people.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'HAVARTH5':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have arthritis'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Did the respondent have arthritis.""")
        st.table(weights)
        st.markdown(""" We see that people with arthritis are twice as more likely to have diabetes compared to people who don't. Previous studies have linked the two to each other and both are considered as risk factors for the other disease.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'CHCKDNY2':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have kidney disease'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Did the respondent have kidney disease.""")
        st.table(weights)
        st.markdown(""" We see that people who had kidney disease were far more likely to have diabetes (over 4.5 times) compared to people who don't. High blood sugar is known to cause damage to kidneys and overtime leads to kidney disease and failure.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'CHCCOPD3':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have COPD'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Did the respondent have COPD (Chronic obstructive pulmonary disease).""")
        st.table(weights)
        st.markdown(""" We see that respondents with COPD were over 2 times as likely to have diabetes compared to those who did not. Both type 1 and type 2 diabetes have been associated with pulmonary complications, and there is a higher prevalence of diabetes occuring in COPD patients.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'ADDEPEV3':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have depressive order'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Did the respondent have a depressive disorder.""")
        st.table(weights)
        st.markdown(""" We see that respondents who had a depressive disorder have a slightly higher chance of having diabetes (~1.4 times) as compared to those who did not.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'CVDSTRK3':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have a stroke'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Did the respondent have a stroke.""")
        st.table(weights)
        st.markdown("""People who have had a stroke were nearly 3 times more likely to have diabetes compared to those who did not. People who have had a stroke are known to have high blood sugars, which tends to damage blood vessels. High blood sugar is also associated with diabetic patients.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'CVDCRHD4':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have coronary heart disease'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Did the respondent have a coronary heart disease.""")
        st.table(weights)
        st.markdown("""People who have a coronary heart disease were 3.6 times more likely to have diabetes compared to those who did not. Patients who have coronary heart disease are known to have high blood pressure and cholesterol, alongside high blood sugar, all risk factors for diabetes.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'CVDINFR4':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have a heart attack'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Did the respondent ever have a heart attack""")
        st.table(weights)
        st.markdown("""People who have had a heart attack were 3.6 times more likely to have diabetes compared to those who did not. Patients who have had heart attacks are known to have high blood pressure and cholesterol, alongside high blood sugar, all risk factors for diabetes.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'BPHIGH6':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Did not have high BP'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Blood pressure status of the respondent.""")
        st.table(weights)
        st.markdown(""" Respondents with high blood pressure were nearly 5 times more likely to have diabetes compared to those who don't. They are also nearly 4 times as likely to have diabetes compared to respondents who had borderline high blood pressure. High blood pressure is associated with insulin resistance which is known to raise blood sugar levels which in turn is a risk factor for diabetes.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'EXERANY2':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Exercised'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""Does the respondent exercise or do physical activity outside of work.""")
        st.table(weights)
        st.markdown(""" People who were physically inactive were nearly 2.5 times more likely to have diabetes compared to those who were active. Physical activity helps control glucose levels, weight, and blood pressure and helps lower cholesterol.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
        
    if assc_select == 'CHECKUP1':
        weights = OddsCalculation(df, assc_select)
        index_val = weights.index[weights['Factor'] == 'Last checkup was less than a year ago'][0]
        weights['Odds Ratio'] = weights['Odds Ratio'] - weights['Odds Ratio'].iloc[index_val]
        weights['Odds Ratio'] = np.exp(weights['Odds Ratio'])
        st.markdown("""When was the last time the respondent had a medical checkup.""")
        st.table(weights)
        st.markdown(""" We see that people who have had infrequent medical checkups were less likely to have diabetes compared to people who had a checkup more recently or have them frequently. People who have regular checkups may have various conditions that need to be monitored which might contribute to diabetes risk. People who have never had a checkup had a higher risk of having diabetes compared to people who had infrequent checkups. Lack of medical checkups may be due to the respondents access to healthcare or inability to use it, thus contributing to a lack of prevention of an intermediary condition leading to a diabetes diagnosis.
        """)
        trend_plot = px.line(weights, x='Factor', y='Odds Ratio', title = 'Associated Risk of getting diabetes')
        st.write(trend_plot)
    