import streamlit as st
import joblib
from utils import preprocess_text

# --- Page Config ---
st.set_page_config(
    page_title="Food Review Analyzer", 
    page_icon="🍽️",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_model()

# --- Predefined Suggestions ---
suggestions = {
    "Positive 😊": "The food was absolutely delicious and the service was excellent! Highly recommend.",
    "Negative 😠": "The food was cold, and the service was incredibly slow. Very disappointed!",
    "Neutral 😐": "The restaurant was okay, nothing too special but not bad either."
}

# --- Callback Functions ---
def update_review(text):
    st.session_state.review_input = text

# --- UI Layout ---
st.title("🍽️ Food Review Sentiment Analyzer")
st.caption("by Sushan Bajracharya | Powered by Linear SVC")

# --- Sidebar ---
with st.sidebar:
    st.warning("""
    **Prototype Limitations**:
    - Trained on ~100K review sample
    - Accuracy: ~73% (Full dataset could improve this)
    - Best for clear positive/negative reviews
    """)

# Initialize session state
if 'review_input' not in st.session_state:
    st.session_state.review_input = ""

# --- Review Input Section ---
st.subheader("Try it out!")
col1, col2 = st.columns([0.7, 0.3])

with col1:
    user_input = st.text_area(
        "Paste your review here:",
        value=st.session_state.review_input,
        height=150,
        placeholder="E.g., 'The pizza was crispy and flavorful...'",
        key="review_input"  
    )

with col2:
    st.markdown("**Or pick a sample:**")
    for label, text in suggestions.items():
        st.button(
            label, 
            on_click=update_review, 
            args=(text,),
            use_container_width=True
        )

# --- Analyze Button ---
if st.button("Analyze Sentiment 🚀", type="primary"):
    if not st.session_state.review_input.strip():
        st.warning("Please enter a review!")
    else:
        with st.spinner("Processing..."):
            try:
                cleaned_text = preprocess_text(st.session_state.review_input)
                vector = tfidf.transform([cleaned_text])
                prediction = model.predict(vector)[0]
                
                st.divider()
                st.subheader("Result")
                if prediction == "positive":
                    st.success(f"✅ **Positive** – The review expresses satisfaction!")
                elif prediction == "negative":
                    st.error(f"❌ **Negative** – The review indicates dissatisfaction.")
                else:
                    st.info(f"🟡 **Neutral** – The review is mixed or indifferent.")
                
            except Exception as e:
                st.error(f"Error processing your review: {str(e)}")

# --- Footer ---
st.divider()
st.markdown("🔍 *Tip: Try reviews with strong emotions for clearer results.*")