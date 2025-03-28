import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



st.header("Diabetes Detection App")

image_path = 'C:\DS\Data Set\diab.png'

try:
    image = Image.open(image_path)
   
    st.image(image, caption='.')
except FileNotFoundError:
    st.error(f"Error: Image file not found at {image_path}")





data = pd.read_csv("C:/DS/Data Set/diabetes.csv")  


st.subheader("data")

st.dataframe(data)

st.subheader("Data Description")

st.write(data.iloc[:,:8].describe())

x = data.iloc[:,:8].values
y = data.iloc[:,8].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model = RandomForestClassifier(n_estimators=500)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

st.subheader("accuracy of train model")
st.write(accuracy_score(y_test,y_pred))

st.subheader("Enter Your Input Data")


def userinput():

    preg = st.slider("Pregnancy",0,20,0)
    glu = st.slider("Glucose",0,200,0)
    bp = st.slider("Blood Pressure",0,130,0)
    sthick = st.slider("Skin Thickness",0,100,0)
    ins = st.slider("Insulin",0.0,1000.0,0.0)
    bmi = st.slider("BMI",0.0,70.0,0.0)
    dpf = st.slider("DPF",0.000,3.000,0.000)
    age = st.slider("Age",0,100,0)

    input_dict = {"Pregnancies":preg, 
    "Glucose":glu, 
    "Blood Pressure":bp, 
    "Skin Thickness":sthick, 
    "Insulin":ins, "BMI":bmi, 
    "DPF":dpf, 
    "Age":age}
    return pd.DataFrame(input_dict,index=["User Input Values"])

ui = userinput()

st.subheader("Entered Input Data")
st.write(ui)


prediction = model.predict(ui)[0]

# Display the result with meaningful text
st.subheader("Prediction Result")
if prediction == 0:
    st.write("The model predicts: **You have No Diabetes** 👍")
else:
    st.write("The model predicts: **You have Diabetes** ⚠️")