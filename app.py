
import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="AI Consumer Behavior Prediction",
    page_icon="🖥️",
    layout="centered"
)

st.title("AI Consumer Behavior Prediction")

st.write(
    "Predict whether consumers are likely to recommend or purchase a product based on review text."
)

review = st.text_area(
    "Enter Product Review",
    placeholder="Example: This product is amazing and worth the price!"
)

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        probability = model.predict_proba(review_vector)[0]

        confidence = max(probability) * 100

        if prediction == 1:
            st.success(
                "Consumers are likely to recommend/purchase this product."
            )
        else:
            st.error(
                "Consumers are unlikely to recommend/purchase this product."
            )

        st.info(
            f"Prediction Confidence: {confidence:.2f}%"
        )
