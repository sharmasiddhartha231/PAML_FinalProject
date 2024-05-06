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

st.markdown("Click **Explore Data** to get started.")