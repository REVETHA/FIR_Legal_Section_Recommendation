import streamlit as st
from transformers import pipeline

# Title
st.title("🚓 Smart FIR - AI Legal Section Recommender")

st.write("Enter incident description to get suggested IPC sections")

# Load model (only once)
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification")

classifier = load_model()

# Input box
text = st.text_area("Enter FIR Description:")

# Button
if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter a description")
    else:
        labels = [
            "IPC 379 - Theft of movable property",
            "IPC 392 - Robbery involving force",
            "IPC 323 - Voluntarily causing hurt",
            "IPC 34 - Acts done by multiple persons (common intention)",
            "IPC 420 - Cheating and fraud",
            "IPC 506 - Criminal intimidation"
            ]

        result = classifier(text, labels)

        st.subheader("📊 Suggested Sections:")

        for label, score in zip(result['labels'], result['scores']):
            st.progress(float(score))
            st.write(f"{label} → {round(score*100, 2)}%")