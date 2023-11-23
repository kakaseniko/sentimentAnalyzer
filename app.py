import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def get_sentiment(text):

    model_name = "SamLowe/roberta-base-go_emotions"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    text = text
    result = emotion_classifier(text)

    emotion_label = result[0]['label']
    return emotion_label

st.write("""
# 🌳 Our Words Matter 📰 🌳
""")

with st.form('my_form'):
  text = st.text_area('Enter text:', '')
  submitted = st.form_submit_button('Submit')
  
  if submitted:
    with st.spinner('Analyzing text...'):
       emotion_label = get_sentiment(text)
       st.info(emotion_label)