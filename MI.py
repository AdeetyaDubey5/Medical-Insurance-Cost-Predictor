import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st #type: ignore

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('/Users/adeetyadubey/Desktop/Project/Medical-Insurance-Cost-Predictor/insurance.csv')
df = data.copy()

# Pre-Processing
le = LabelEncoder()

df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

# Splitting the train-test data

X = df.drop('charges', axis=1)
y = df['charges']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Creating the model
model = LinearRegression()
model.fit(X_train,y_train)

# Creating the Front-End

st.sidebar.title('Sections')
section = st.sidebar.radio('Go to',['Home','Data Analysis','About'])


#Home
if section=='Home':

    st.title('Medical Insurance Cost Predictor')
    st.subheader('Enter details to estimate your insurance cost:')

    #Inputs
    age = st.slider('Age',1,100)
    sex = st.selectbox('Sex',('Male','Female'))
    bmi = st.number_input('BMI',1.00,42.00,format='%.2f')
    children = st.slider('Children',0,5)
    smoker = st.selectbox('Smoker',('Yes','No'))
    region = st.selectbox('Region',('Northeast','Northwest','Southeast','Southwest'))

    def region(n):
        if n == 'northeast':
            return 0
        elif n == 'northwest':
            return 1
        elif n == 'southeast':
            return 2
        else:
            return 3

    #Converting Inputs to Dataframe
    input = pd.DataFrame({
        'age': [age],
        'sex': [0 if sex == 'Male' else 1],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == 'Yes' else 0],
        'region': [region(region)]
    })

    # Making Prediction
    if st.button('Predict'):
        prediction = model.predict(input)
        st.write(f"Estimate Insurance Cost: ₹{prediction[0]: .2f}")
       

#Data Analysis
elif section=='Data Analysis':
    st.title('Data Analysis')
    st.write('')
    st.write('')

    #Charges vs frequency
    st.header('1. Distribution of Insurance Charges')
    fig = plt.figure(figsize=(10,6))
    sns.histplot(df['charges'],kde=True,bins=30)
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    st.pyplot(fig)
    st.subheader('Insights:')
    st.markdown('- Skewness')
    st.write('The distribution is right-skewed. This indicates that most policyholders have lower insurance charges, but there are a few with significantly higher charges.')
    st.markdown('- Density')
    st.write('The KDE (Kernel Density Estimate) curve provides a smooth estimate of the distribution. The peak of the KDE line reinforces that the majority of the data is concentrated at lower charge values.')

    #Age vs Charges
    st.header('2. Insurance Charges by Age')
    fig = plt.figure(figsize=(10,6))
    sns.scatterplot(data=data,x='age',y='charges',hue='sex')
    plt.title('Insurance Charges by Age')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    st.pyplot(fig)
    st.subheader('Insights:')
    st.markdown('- General trend')
    st.write('There is a general upward trend in charges as age increases, indicating that older individuals tend to have higher insurance charges.')
    st.markdown('- Outliers')
    st.write('There are notable outliers with extremely high charges, particularly at younger ages (around 20-30). These could be cases of significant medical expenses that greatly exceed the average.')
    st.write('')
    st.write('Additionally, there is no clear, consistent difference in charges between the sexes at any given age. Both sexes seem to follow a similar distribution pattern.')

    #Sex vs Charges
    st.header('3. Insurance Charges by Sex')
    fig = plt.figure(figsize=(10,6))
    sns.barplot(data=data,x='sex',y='charges',hue='smoker')
    plt.title('Insurance Charges by Sex')
    plt.xlabel('Sex')
    plt.ylabel('Charges')
    st.pyplot(fig)
    st.subheader('Insights:')
    st.markdown('- General trend')
    st.write('While sex does have some impact on insurance charges, it is less pronounced compared to the impact of smoking.')
    st.markdown('- Smoking status')
    st.write('Both male and female smokers have significantly higher insurance charges compared to non-smokers.')
    st.write('\nSmoking status is a major determinant of insurance charges, with smokers incurring much higher costs.')

    #BMI vs Charges
    st.header('4. Insurance Charges by BMI')
    fig = plt.figure(figsize=(10,6))
    sns.scatterplot(data=data,x='bmi',y='charges',hue='smoker')
    plt.title('Insurance Charges by BMI')
    plt.xlabel('BMI')
    plt.ylabel('Charges')
    st.pyplot(fig)
    st.subheader('Insights:')
    st.markdown('- Positive Correlation Between BMI and Charges')
    st.write('There is a general positive correlation between BMI and insurance charges. As BMI increases, the insurance charges also tend to increase.')
    st.markdown('- Clustering of Data Points')
    st.write('There is a dense clustering of data points in the BMI range of 20-30 with varying charges. This indicates that most individuals fall within this BMI range and have a wide range of insurance charges.')

    #Children vs Charges
    st.header('5. Insurance Charges by Children')
    fig = plt.figure(figsize=(10,6))
    sns.barplot(data=data,x='children',y='charges',hue='smoker')
    plt.title('Insurance Charges by Children')
    plt.xlabel('Children')
    plt.ylabel('Charges')
    st.pyplot(fig)
    st.subheader('Insights:')
    st.markdown('- General trend')
    st.write ('There is no clear, consistent pattern in insurance charges as the number of children increases.\nCharges do not systematically increase or decrease with more children.')
    st.markdown('- Variation in Charges')
    st.write('For families with 4 children, the variation (represented by the error bars) is quite large, indicating a wide range of insurance charges.For families with 5 children, both the mean and variation of charges decrease.')

    #Smoker vs Charges
    st.header('6. Insurance Charges by Smoking Status')
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='smoker', y='charges')
    plt.title('Charges by Smoker')
    plt.xlabel('Smoker')
    plt.ylabel('Charges')
    st.pyplot(fig)
    st.subheader('Insights:')
    st.markdown('- Median Charges')
    st.write('For smokers, the median charge is higher, exceeding Rs 30,000. Non-smokers have a significantly lower median charge.')
    st.markdown('- Variability')
    st.write('The interquartile range (IQR) for smokers is larger, indicating more variability in charges among smokers. Non-smokers exhibit a narrower IQR, suggesting less variability.')

    #Region vs Charges
    st.header('7. Insurance Charges by Region')
    fig = plt.figure(figsize=(10,6))
    sns.barplot(data=data,x='region',y='charges',hue='smoker')
    plt.title('Insurance Charges by Region')
    plt.xlabel('Region')
    plt.ylabel('Charges')
    st.pyplot(fig)
    st.subheader('Insights:')
    st.markdown('- General trend')
    st.write('The insurance charges for both the northern and southern regions are quite similar, with the southern regions showing slightly higher charges than the northern regions.')
    st.markdown('- Regional Variation')
    st.write('The graph shows that insurance charges are highest for smokers in the southeast region, while for non-smokers, the highest charges are in the northeast region.')

elif section=='About':
    st.header("About")

    st.subheader("Overview")
    st.write("""
    This project aims to predict the average medical insurance cost of an individual based on various features such as age, sex, BMI, number of children, smoking status, and region. The dataset used for this analysis is a comprehensive collection of medical insurance charges along with the associated features.
    """)

    st.subheader("Model and Methodology")
    st.write("""
    A *Linear Regression* model was used to predict the average medical insurance cost. Linear regression is a simple yet powerful statistical method for predicting a quantitative response. The features mentioned above were used to train the model and make predictions.
    """)

    st.subheader("Conclusion")
    st.write("""
    This project provides valuable insights into the factors affecting medical insurance costs and helps in predicting these costs more accurately. The **Linear Regression** model built in this project can serve as a foundational tool for insurance companies to better understand their pricing models and for individuals to estimate their potential medical insurance expenses.
    """)

    st.subheader("Author")
    st.write("**Adeetya Dubey**")
    st.write("""
    This project was Developed by a passionate data scientist dedicated to leveraging data analysis and machine learning techniques to solve real-world problems. Adeetya's expertise in predictive modeling and statistical analysis has been instrumental in creating this insightful and practical tool for predicting medical insurance costs.
    """)
    