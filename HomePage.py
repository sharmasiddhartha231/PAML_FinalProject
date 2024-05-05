import streamlit as st               

st.markdown("# Predicting Diabetes and investigating Factors associated with Diabetes Risk")

#############################################

st.markdown("This is the application for Final Project for Practical Applications of Machine Learning. In this application, we will be exploring a publicly available dataset to predict whether an adult has Diabetes or not and studying the assocation of different factors with risk of diabetes. The application also provides the users the ability to train the models themselves using various parameters and factors. Finally, we provide an interface where the user can enter their health details and see the results of the predictions based on the models we have trained.")

st.markdown("""Diabetes is one of the most prevalent chronic diseases in the United States, with over a million cases annually and an estimated 38.4 million cases in total in the United States. Amongst these, Type 2 Diabetes is the most prevalent form and it accounts for nearly 95% of the total Diabetes cases. Prediabetes is a condition which can lead to Type 2 Diabetes. Approximately one third of the population of the United States has prediabetes and more than 80% of it goes undiagnosed. With our application, we aim to predict whether a user has diabetes or not using various factors and investigate the association of these factors to increased risk of diabetes and prediabetes. We will be using a publicly available dataset published by CDC called the BRFSS data which consists of information on various metrics collected annually.

With our application we aim to:
- Allow the users to explore the various factors used for prediction.
- Build end-to-end classification pipeline using 1) logistic regression, 2) K Nearest Neighbors, and 3) Random Forest Classifiers.
- Evaluate these methods using standard metrics including Accuracy, Precision, Recall and others.
- Provide the collated results of the pipeline highlighting the methods which work best and detail the various risk factors for diabetes.
- Develop a web application that walks users through steps of the  pipeline and provide tools to analyze multiple methods across multiple metrics. 
- Provide an interface for the users to enter their input data and predict if they have diabetes or not.
""")

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

st.markdown("Click **Explore Data** to get started.")