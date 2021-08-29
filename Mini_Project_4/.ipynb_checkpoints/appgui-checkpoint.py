import pickle
import requests
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

st.set_page_config(layout="centered")

st.title('Mortgage Pre-Approval Tool')
st.markdown('Find some **_Peace of Mind_**')
st.write("---")
st.markdown("Are you in need of a loan? Use this tool to put your mind at ease before beginning the stressful application process.")
st.markdown("Check out this plot as a general overview of loan applications for comparison.")

df = pd.read_csv('data.csv')
demo_data = df.iloc[:300,:].copy()
demo_data = demo_data[(demo_data['ApplicantIncome'] <= 12000) & (demo_data['LoanAmount'] < 350)]

st.write("")
data_plot = alt.Chart(demo_data).mark_circle().encode(x='LoanAmount', y='ApplicantIncome', color='Loan_Status',
                                                          tooltip=['LoanAmount', 'ApplicantIncome'])
st.write(data_plot)
st.write("---")

model = pickle.load(open("loan_prediction.pickle", "rb"))

# prediction function that will use the data which the user inputs 
def prediction(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):

    if Married == "Unmarried":
        Married = 'No'
    else:
        Married = 'Yes'

    if Credit_History == "I have credit":
        Credit_History = 1
    else:
        Credit_History = 0  
 
    LoanAmount = LoanAmount / 1000
    # Format input data for predictions
    df = pd.DataFrame([{'Gender': Gender,
                        'Married': Married,
                        'Dependents': Dependents,
                        'Education': Education,
                        'Self_Employed': Self_Employed,
                        'ApplicantIncome': ApplicantIncome,
                        'CoapplicantIncome': CoapplicantIncome,
                        'LoanAmount': LoanAmount,
                        'Loan_Amount_Term': Loan_Amount_Term,
                        'Credit_History': Credit_History,
                        'Property_Area': Property_Area}])
    # Making predictions
    prediction = model.predict(df)

    if prediction[0] == 'N':
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred

st.subheader("**Now try it for yourself!**")
st.markdown("Fill out the form below and make a prediction for your mortgage approval!")
# following lines create boxes in which user can enter data required to make prediction 
Gender = st.selectbox('Gender',("Male","Female"))
Married = st.selectbox('Marital Status',("Unmarried","Married"))
Dependents = st.selectbox('Dependents', ("0","1","2","3+"))
Education = st.selectbox('Education', ("Graduate","Not Graduate"))
Self_Employed = st.selectbox('Self Employed', ("Yes","No"))
ApplicantIncome = st.number_input("Applicants monthly income")
CoapplicantIncome = st.number_input("Coapplicants monthly income, if applicable")
LoanAmount = st.number_input("Total loan amount")
Loan_Amount_Term = st.number_input("Loan Term (months)")
Credit_History = st.selectbox('Credit History',("I have credit","I do not have credit"))
Property_Area = st.selectbox('Property Class', ("Urban","Semi Urban","Rural"))
result =""

# when 'Predict' is clicked, make the prediction and store it 
if st.button("Predict"): 
    result = prediction(Gender, Married, Dependents, Education,
                        Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
                        Loan_Amount_Term, Credit_History, Property_Area)
    if result == 'Approved':
        st.success(':sunglasses: Your loan had a good chance to be {}'.format(result))
    else:
        st.write('😩 Sorry, it looks like your application will be denied.')
    # print value to terminal after successful prediction
    print(LoanAmount)