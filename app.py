import streamlit as st
import pickle
import numpy as np

# Set page layout and title
st.set_page_config(page_title="Student Mental Health Prediction", layout="wide", page_icon="🧠")
st.title("🔍 Student Mental Health Prediction")
st.subheader("Find out if a student may be experiencing depression based on their profile.")

# Load the trained model
model_path = 'RandomForestModel.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Sidebar with additional options
with st.sidebar:
    st.image("mental_health_banner.png", use_column_width=True)  # You can use a custom image here
    st.header("About")
    st.write(
        """
        This application uses a trained Machine Learning model to predict the likelihood of depression among students.
        Fill in the details below to get an instant assessment.
        """
    )
    st.write("Developed by: U-Connect")

# Collecting user input
st.markdown("### Enter your details:")
gender = st.selectbox("Gender", ("Male", "Female"))
age = st.slider("Age", 18, 30, 20)
year = st.selectbox("Year of Study", ("1", "2", "3", "4"))
cgpa = st.selectbox("CGPA Range", ('0', '1', '2', '3', '4'))
marriage = st.selectbox("Marital Status", ("No", "Yes"))
anxiety = st.selectbox("Do you have Anxiety?", ("No", "Yes"))
panic = st.selectbox("Do you have Panic Attack?", ("No", "Yes"))
treatment = st.selectbox("Have you sought Treatment?", ("No", "Yes"))
course = st.text_input("Enter your course (e.g., Engineering, Psychology)")

# Converting inputs to the expected format
gender = 1 if gender == 'Male' else 0
marriage = 0 if marriage == 'Yes' else 1
anxiety = 0 if anxiety == 'Yes' else 1
panic = 0 if panic == 'Yes' else 1
treatment = 0 if treatment == 'Yes' else 1

# Prepare input vector
input_data = np.zeros(len(model.feature_importances_))
input_data[0] = gender
input_data[1] = age
input_data[2] = int(year)
input_data[3] = int(cgpa)
input_data[4] = marriage
input_data[5] = anxiety
input_data[6] = panic
input_data[7] = treatment

# Checking if course exists in the model's feature columns
course_feature = "Course_" + course
if course_feature in model.feature_importances_:
    course_index = np.where(model.feature_importances_ == course_feature)[0][0]
    input_data[course_index] = 1

# Prediction
if st.button("Predict"):
    result = model.predict([input_data])[0]
    if result == 0:
        st.success("The student is likely experiencing Depression.")
    else:
        st.success("The student is not likely experiencing Depression.")

# Add footer with social media links
st.markdown("---")
st.markdown("💬 Connect with us: [LinkedIn](https://www.linkedin.com/) | [Twitter](https://twitter.com/) | [GitHub](https://github.com/)")

# Style adjustments using HTML/CSS
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        color: #333;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 10px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
