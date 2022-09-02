import streamlit as st
from pytube import extract
from googleapiclient.discovery import build
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


@st.cache(allow_output_mutation=True)
def load_models():
    print("loading model")
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment"
    )

    return tokenizer, model


tokenizer, model = load_models()
api_key = "AIzaSyADHkcRKOWROU5HF4T3C_cC54OPfeoqdSE"
service = build("youtube", "v3", developerKey=api_key)


def get_video_id(url):
    return extract.video_id(url)


def clean_txt(text):
    text = text.lower()
    text = re.sub(
        r"(@\[A-Za-z0-9] )|([^0-9A-Za-z \t])|(\w :\/\/\S )|^rt|http. ?", "", text
    )
    return text


def get_comments(video_id):
    req = service.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=100
    )

    response = req.execute()

    comments = []
    real = []

    for i in response["items"]:
        cmt = i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        real.append(cmt)
        cmt = clean_txt(cmt)
        comments.append(cmt)

    return comments, real


def predict(comments, real):
    for i in range(len(comments)):
        encoded_input = tokenizer(comments[i], return_tensors="pt")
        output = model(**encoded_input)
        output = torch.argmax(output[0][0])

        if output == 0:
            st.error(real[i])

        elif output == 1:
            st.warning(real[i])

        else:
            st.success(real[i])


def show(url):
    url = get_video_id(url)
    comments, real = get_comments(url)
    predict(comments, real)


st.title("  YOUTUBE COMMENTS FLAGGER")

col1, col2, col3 = st.columns(3)
with col1:
    st.error("NEGATIVE")
with col2:
    st.warning("NEUTRAL")
with col3:
    st.success("POSITIVE")


url = st.text_input("Enter the youtube video url")

if st.button("SUBMIT"):
    show(url)
