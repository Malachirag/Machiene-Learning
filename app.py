import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Function for crop recommendation
def recommendation(N,P,k,temperature,humidity,ph,rainfall):
    features = np.array([[N,P,k,temperature,humidity,ph,rainfall]])
    prediction = model.predict(features).reshape(1,-1)
    return prediction[0][0]

# Create a Streamlit web application
st.title("Crop Recommendation System")

# Sidebar with user inputs
st.sidebar.header("User Input")

# Get user inputs
N = st.sidebar.number_input("Nitrogen (N)", 1, 100, 40)
P = st.sidebar.number_input("Phosphorous (P)", 1, 100, 50)
k = st.sidebar.number_input("Potassium (K)", 1, 100, 50)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 100.0, 40.0)
humidity = st.sidebar.number_input("Humidity (%)", 0, 100, 20)
ph = st.sidebar.number_input("pH Level", 0, 14, 7)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0, 1000, 100)

# Button to get recommendations
if st.sidebar.button("Get Recommendation"):
    prediction = recommendation(N, P, k, temperature, humidity, ph, rainfall)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    if prediction in crop_dict:
        crop = crop_dict[prediction]
        st.success(f"The best crop to be cultivated is {crop}.")
    else:
        st.error("Sorry, we are not able to recommend a proper crop for this environment.")
