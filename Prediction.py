import streamlit as st
import joblib
import re

st.set_page_config(page_title="AuthentiScan - A Text Classifier", page_icon="🤖")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

@st.cache_resource
def load_model():
    model = joblib.load("logreg_ai_human_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_text(text):
    clean = clean_text(text)
    vect = vectorizer.transform([clean])
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0]
    label_name = "AI" if pred == 1 else "Human"
    confidence = prob[pred]
    return label_name, confidence

example_texts = [
    "Artificial intelligence is transforming industries across the globe with unprecedented speed and efficiency.",
    "This paper investigates the implications of deep reinforcement learning in multi-agent systems.",
    "I went on a walk this morning, watched the sun rise, and felt truly at peace with everything.",
    "The camera quality is fantastic, and the battery lasts me an entire day. Totally worth the price!",
    "Unlock your business potential with our all-in-one cloud solution that drives performance and scalability."
]

if "example_index" not in st.session_state:
    st.session_state.example_index = 0
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

st.markdown("<div class='main-title'>🤖 AuthentiScan - A Text Classifier</div>", unsafe_allow_html=True)
st.markdown("Paste or type any text below to predict whether it’s written by an **AI** or a **Human**.")

if st.button("Try Example Text"):
    st.session_state.text_input = example_texts[st.session_state.example_index]
    st.session_state.example_index = (st.session_state.example_index + 1) % len(example_texts)

text_default = st.session_state.get("text_input", "")
user_input = st.text_area("Text Input", value=text_default, height=180)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence = predict_text(user_input)
        if(label == "AI" and confidence > 0.96):
            st.markdown(f"""
            <div class='prediction-box fade-in'>
                <h4>Prediction: <span style="color:#2196f3">{label}</span></h4>
                <p>Confidence: <strong>{confidence:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='prediction-box fade-in'>
                <h4>Prediction: <span style="color:#2196f3">{"Human"}</span></h4>
                <p>Confidence: <strong>{confidence:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
