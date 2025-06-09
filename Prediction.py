# import streamlit as st
# import joblib
# import re

# st.set_page_config(page_title="AI vs Human Text Classifier", page_icon="🤖")

# # === 1. Load model and vectorizer ===
# @st.cache_resource
# def load_model():
#     model = joblib.load("logreg_ai_human_model.joblib")
#     vectorizer = joblib.load("tfidf_vectorizer.joblib")
#     return model, vectorizer

# model, vectorizer = load_model()

# # === 2. Text cleaning function ===
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"[^a-z0-9\s]", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# # === 3. Prediction function ===
# def predict_text(text):
#     clean = clean_text(text)
#     vect = vectorizer.transform([clean])
#     pred = model.predict(vect)[0]
#     prob = model.predict_proba(vect)[0]
#     label_name = "AI" if pred == 1 else "Human"
#     confidence = prob[pred]
#     return label_name, confidence

# # === 4. Streamlit UI ===
# # st.set_page_config(page_title="AI vs Human Text Classifier", page_icon="🤖")

# st.title("🤖 AI vs Human Text Classifier")
# st.markdown("Enter any text below and find out whether it was likely written by an **AI** or a **Human**.")

# user_input = st.text_area("Enter your text here:", height=200)

# if st.button("Classify"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         label, confidence = predict_text(user_input)
#         st.subheader("Prediction Result")
#         st.markdown(f"**Prediction**: {label}")
#         st.markdown(f"**Confidence**: {confidence:.2%}")


# import streamlit as st
# import joblib
# import re
# from pathlib import Path

# # === Page setup ===
# st.set_page_config(page_title="AuthentiScan Text Classifier", page_icon="🤖")

# # === Load CSS ===
# def load_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# load_css("style.css")

# # === Load model and vectorizer ===
# @st.cache_resource
# def load_model():
#     model = joblib.load("logreg_ai_human_model.joblib")
#     vectorizer = joblib.load("tfidf_vectorizer.joblib")
#     return model, vectorizer

# model, vectorizer = load_model()

# # === Text cleaner ===
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"[^a-z0-9\s]", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# # === Predictor ===
# def predict_text(text):
#     clean = clean_text(text)
#     vect = vectorizer.transform([clean])
#     pred = model.predict(vect)[0]
#     prob = model.predict_proba(vect)[0]
#     label_name = "AI" if pred == 1 else "Human"
#     confidence = prob[pred]
#     return label_name, confidence

# # === UI ===
# st.markdown("<div class='main-title'>🤖 AuthentiScan Text Classifier</div>", unsafe_allow_html=True)
# st.markdown("Paste or type any text below to predict whether it’s written by an **AI** or a **Human**.")

# # Example text button
# example_text = "Artificial intelligence is transforming industries across the globe."
# if st.button("Try Example Text"):
#     st.session_state['text_input'] = example_text

# # Text area with session state
# text_default = st.session_state.get('text_input', '')
# user_input = st.text_area("Text Input", value=text_default, height=180)

# # Prediction
# if st.button("Classify"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         label, confidence = predict_text(user_input)
#         st.markdown(f"""
#         <div class='prediction-box'>
#             <h4>Prediction: <span style="color:#2196f3">{label}</span></h4>
#             <p>Confidence: <strong>{confidence:.2%}</strong></p>
#         </div>
#         """, unsafe_allow_html=True)


import streamlit as st
import joblib
import re

# === Page setup ===
st.set_page_config(page_title="AuthentiScan - A Text Classifier", page_icon="🤖")

# === Load custom CSS ===
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# === Load model and vectorizer ===
@st.cache_resource
def load_model():
    model = joblib.load("logreg_ai_human_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model()

# === Clean input text ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Predict function ===
def predict_text(text):
    clean = clean_text(text)
    vect = vectorizer.transform([clean])
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0]
    label_name = "AI" if pred == 1 else "Human"
    confidence = prob[pred]
    return label_name, confidence

# === Example texts ===
example_texts = [
    "Artificial intelligence is transforming industries across the globe with unprecedented speed and efficiency.",
    "This paper investigates the implications of deep reinforcement learning in multi-agent systems.",
    "I went on a walk this morning, watched the sun rise, and felt truly at peace with everything.",
    "The camera quality is fantastic, and the battery lasts me an entire day. Totally worth the price!",
    "Unlock your business potential with our all-in-one cloud solution that drives performance and scalability."
]

# === Session state setup ===
if "example_index" not in st.session_state:
    st.session_state.example_index = 0
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# === Title and Instructions ===
st.markdown("<div class='main-title'>🤖 AuthentiScan - A Text Classifier</div>", unsafe_allow_html=True)
st.markdown("Paste or type any text below to predict whether it’s written by an **AI** or a **Human**.")

# === Try Example Text Button ===
if st.button("Try Example Text"):
    st.session_state.text_input = example_texts[st.session_state.example_index]
    st.session_state.example_index = (st.session_state.example_index + 1) % len(example_texts)

# === Text input ===
text_default = st.session_state.get("text_input", "")
user_input = st.text_area("Text Input", value=text_default, height=180)

# === Classify button ===
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence = predict_text(user_input)
        st.markdown(f"""
        <div class='prediction-box fade-in'>
            <h4>Prediction: <span style="color:#2196f3">{label}</span></h4>
            <p>Confidence: <strong>{confidence:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

