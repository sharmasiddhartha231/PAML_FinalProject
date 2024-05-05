import streamlit as st               

st.markdown("# Predicting Diabetes and investigating Factors associated with Diabetes Risk")

#############################################

st.markdown("This is the application for Final Project for Practical Applications of Machine Learning. In this application, we will be exploring a publicly available dataset to predict whether an adult has Diabetes or not and studying the assocation of different factors with risk of diabetes. The application also provides the users the ability to train the models themselves using various parameters and factors. Finally, we provide an interface where the user can enter their health details and see the results of the predictions based on the models we have trained.")

st.markdown("""Diabetes is one of the most prevalent chronic diseases in the United States, with over a million cases annually and an estimated 38.4 million cases in total in the United States. Amongst these, Type 2 Diabetes is the most prevalent form and it accounts for nearly 95%% of the total Diabetes cases. Prediabetes is a condition which can lead to Type 2 Diabetes. Approximately one third of the population of the United States has prediabetes and more than 80%% of it goes undiagnosed. 

The learning outcomes for this assignment are:
- Build end-to-end regression pipeline using 1) multiple regression, 2) polynomial regression, and 3) ridge regression.
- Evaluate regression methods using standard metrics including root mean squared error (RMSE), mean absolute error (MAE), and coefficient of determination (R2).
- Develop a web application that walks users through steps of the regression pipeline and provide tools to analyze multiple methods across multiple metrics. 
""")

st.markdown(""" California Housing Data

This assignment involves testing the end-to-end pipeline in a web application using a California Housing dataset from the textbook: Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. O’Reilly Media, Inc., 2022 [GitHub,Dataset Description]. The dataset was captured from California census data in 1990. We have added additional features to the dataset. The features include:
- longitude: longitudinal coordinate
- latitude: latitudinal coordinate
- housing_median_age: median age of district
- total_rooms: total number of rooms per district
- total_bedrooms: total number of bedrooms per district
- population: total population of district
- households: total number of households per district'
- median_income: median income
- ocean_proximity: distance from the ocean
- median_house_value: median house value
- city: city location of house
- county: county of house
- road: road of the house
- postcode: zip code 

""")

st.markdown("Click **Explore Data** to get started.")