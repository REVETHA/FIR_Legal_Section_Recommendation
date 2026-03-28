import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(page_title="Smart FIR AI", page_icon="🚓", layout="centered")

# Title
st.title("🚓 Smart FIR - AI Legal Section Recommender")
st.write("Enter incident description to get suggested IPC sections")

# Load model (cached)
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification")

classifier = load_model()

# IPC Labels
labels = [
    "IPC 379 - Theft of movable property",
    "IPC 392 - Robbery involving force",
    "IPC 323 - Voluntarily causing hurt",
    "IPC 34 - Acts done by multiple persons (common intention)",
    "IPC 420 - Cheating and fraud",
    "IPC 506 - Criminal intimidation"
]

# IPC Explanations
ipc_explanations = {
    "IPC 379 - Theft of movable property": "Punishment for theft of property.",
    "IPC 392 - Robbery involving force": "Theft combined with violence or threat.",
    "IPC 323 - Voluntarily causing hurt": "Causing physical injury intentionally.",
    "IPC 34 - Acts done by multiple persons (common intention)": "Crime done by multiple people with shared intent.",
    "IPC 420 - Cheating and fraud": "Dishonest inducement to gain property.",
    "IPC 506 - Criminal intimidation": "Threatening a person with harm."
}

# Input
text = st.text_area("Enter FIR Description:")

# Button
if st.button("Analyze"):

    if text.strip() == "":
        st.warning("⚠️ Please enter a description")
    else:
        result = classifier(text, labels)

        # Sort results
        sorted_results = sorted(
            zip(result['labels'], result['scores']),
            key=lambda x: x[1],
            reverse=True
        )

        # Top Prediction
        top_label, top_score = sorted_results[0]

        st.success(
            f"✅ Most Relevant Section: {top_label} ({round(top_score*100, 2)}%)"
        )

        st.subheader("📊 AI Suggested Legal Sections")

        # Display all predictions
        for label, score in sorted_results:
            st.progress(float(score))
            st.write(f"{label} → {round(score*100, 2)}%")
            st.caption(ipc_explanations.get(label, ""))

# Footer
st.markdown("---")
st.caption("⚠️ This is an AI-assisted recommendation. Final decision should be verified by the officer.")