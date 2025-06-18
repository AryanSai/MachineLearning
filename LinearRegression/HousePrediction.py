import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def generate_house_data(num_houses = 100):
    np.random.seed(50)
    size = np.random.normal(1400, 50, num_houses) #mean, sd, num_of_samples
    price = size * 50 + np.random.normal(0, 50, num_houses)
    return pd.DataFrame({'Size': size, 'Price': price})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model():
    df = generate_house_data()
    X = df[['Size']]
    y = df[['Price']]
    X_train, X_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def main():
    st.title("House Price Prediction App")
    st.write("Enter the size of the house in square feet to predict its price.")

    model = train_model()
    size = st.number_input("Size of the house (in square feet)", min_value=500, max_value=2000, value=1500)

    if st.button("Predict Price"):
        predicted_price = model.predict([[size]])
        st.success(f"The predicted price for a house of size {size} sq ft is ${predicted_price[0][0]:.2f}")

        df = generate_house_data()

        fig = px.scatter(df, x='Size', y='Price', title="House Price vs Size")
        fig.add_scatter(x=[size], y=[predicted_price[0][0]], mode='markers', marker=dict(color='red', size=10), name='Predicted Price')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()