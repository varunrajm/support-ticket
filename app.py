import streamlit as st
import time
from src.predict import load_models, classify_ticket

st.set_page_config(
    page_title="Ticket Auto-Classifier", 
    page_icon="🎫", 
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
        border-left: 5px solid #2e6c80;
    }
    .action-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎫 Support Ticket Auto-Classifier")
st.markdown("Enter a customer support ticket below to automatically classify its **Category** and **Priority** using our Machine Learning pipeline.")

# Load models with a spinner
@st.cache_resource(show_spinner="Loading NLP Models...")
def init_models():
    return load_models()

try:
    cat_model, prio_model, cat_enc, prio_enc = init_models()
except Exception as e:
    st.error("Error loading models. Please ensure you have run the training pipeline first (`python run_all.py`).")
    st.exception(e)
    st.stop()

# Input Form
with st.form("ticket_form"):
    subject = st.text_input("Ticket Subject", placeholder="e.g., App crashes on startup after the latest update")
    description = st.text_area("Ticket Description", placeholder="e.g., Every time I open the app on my iPhone, it immediately closes. I have tried reinstalling it but it still doesn't work. I have a presentation tomorrow and this is urgent.", height=150)
    
    submitted = st.form_submit_button("Classify Ticket", type="primary")

if submitted:
    if not subject.strip() and not description.strip():
        st.warning("Please provide a subject or a description to classify.")
    else:
        with st.spinner("Analyzing ticket context and urgency..."):
            time.sleep(0.5) # Simulate slight processing time for effect
            res = classify_ticket(subject, description, cat_model, prio_model, cat_enc, prio_enc)
            
            st.markdown("### Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Category", res['category'], f"{res['category_confidence']:.0%} confidence", delta_color="off")
            with col2:
                # Color code priority
                p_color = "red" if res['priority'] == "High" else "orange" if res['priority'] == "Medium" else "green"
                st.metric("Priority", res['priority'], f"{res['priority_confidence']:.0%} confidence", delta_color="off")
                
            st.markdown(f"""
            <div class="action-card">
                <strong>Recommended Action:</strong><br/>
                {res['recommended_action']}
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Machine Learning Internship Project • End-to-End Classification System*")
