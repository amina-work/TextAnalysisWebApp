from tracemalloc import stop
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")

# Custom CSS for st.text_area
with st.sidebar:
    rad = option_menu(
        menu_title="Navigation",
        options=["Home","Spam or Ham Detection","Sentiment Analysis","Stress Detection","Hate and Offensive Content Detection","Sarcasm Detection"],
        icons=["house", "bi bi-arrow-right-circle", "bi bi-arrow-right-circle", "bi bi-arrow-right-circle", "bi bi-arrow-right-circle", "bi bi-arrow-right-circle"]
    )
#Home Page
if rad=="Home":
    st.title("Complete Text Analysis App")
    #st.image("Complete Text Analysis Home Page.jpg")
    st.image("pic1.jpg")
    st.text(" ")
    st.write(
        "The Following Text Analysis Options Are Available ⇒",
    )

    st.markdown("1. **Spam or Ham Detection**:")
    st.write("Is the text genuine or potential spam?")

    st.markdown("2. **Sentiment Analysis**:")
    st.write("Understand the underlying sentiment of the text.")

    st.markdown("3. **Stress Detection**:")
    st.write("Determine stress levels from the text's tone and content.")

    st.markdown("4. **Hate and Offensive Content Detection**:")
    st.write("Identify any hateful or offensive content in the text.")

    st.markdown("5. **Sarcasm Detection**:")
    st.write("A unique feature that helps detect sarcasm in the text.")

    st.text("")
    st.text("")
    st.text("")
    st.subheader("The Links For the Text Analysis Options")

#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#Spam Detection Prediction
tfidf1=TfidfVectorizer(stop_words=sw,max_features=20)
def transform1(txt1):
    txt2=tfidf1.fit_transform(txt1)
    return txt2.toarray()

df1=pd.read_csv("Spam Detection.csv")
df1.columns=["Label","Text"]
x=transform1(df1["Text"])
y=df1["Label"]
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.1,random_state=0)
model1=LogisticRegression()
model1.fit(x_train1,y_train1)

#Background color for explanation texts
custom_style = """
    <style>
        .custom-text {
            background-color: #ff6800;
            padding:5px 10px;
            border-radius: 5px;
            color: #ffffff;
            display: inline-block;
        }
    </style>
    """
st.markdown(custom_style, unsafe_allow_html=True)

# CSS for buttons
st.write("""
<style>
    .stButton button {
        color: #ffffff;
        background-color: #ff6800;
    }
</style>
""", unsafe_allow_html=True)
#Spam Detection Analysis Page
if rad=="Spam or Ham Detection":
    st.header("Detect Whether A Text Is Spam Or Ham??")
    # Display text with the custom style
    st.text("")
    st.markdown('<div class="custom-text">Is the text genuine or potential spam?</div>', unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.write("""
        <style>
            textarea {
                background-color: #ffffff !important;
            }
        </style>
        """, unsafe_allow_html=True)
    sent1 = st.text_area("Enter The Text")
    transformed_sent1 = transform_text(sent1)
    vector_sent1 = tfidf1.transform([transformed_sent1])
    prediction1 = model1.predict(vector_sent1)[0]

    if st.button("Predict"):
        if prediction1=="spam":
            st.warning("Spam Text!!")
        elif prediction1=="ham":
            st.success("Ham Text!!")

#Sentiment Analysis Prediction 
tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
def transform2(txt1):
    txt2=tfidf2.fit_transform(txt1)
    return txt2.toarray()

df2=pd.read_csv("Sentiment Analysis.csv")
df2.columns=["Text","Label"]
x=transform2(df2["Text"])
y=df2["Label"]
x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.1,random_state=0)
model2=LogisticRegression()
model2.fit(x_train2,y_train2)

#Sentiment Analysis Page
if rad=="Sentiment Analysis":
    st.header("Detect The Sentiment Of The Text!!")
    st.text("")
    st.markdown('<div class="custom-text">Understand the underlying sentiment of the text.</div>', unsafe_allow_html=True)
    st.text("")
    st.text("")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent2=tfidf2.transform([transformed_sent2])
    prediction2=model2.predict(vector_sent2)[0]

    if st.button("Predict"):
        if prediction2==0:
            st.warning("Negetive Text!!")
        elif prediction2==1:
            st.success("Positive Text!!")

#Stress Detection Prediction
tfidf3=TfidfVectorizer(stop_words=sw,max_features=20)
def transform3(txt1):
    txt2=tfidf3.fit_transform(txt1)
    return txt2.toarray()

df3=pd.read_csv("Stress Detection.csv")
df3=df3.drop(["subreddit","post_id","sentence_range","syntax_fk_grade"],axis=1)
df3.columns=["Text","Sentiment","Stress Level"]
x=transform3(df3["Text"])
y=df3["Stress Level"].to_numpy()
x_train3,x_test3,y_train3,y_test3=train_test_split(x,y,test_size=0.1,random_state=0)
model3=DecisionTreeRegressor(max_leaf_nodes=2000)
model3.fit(x_train3,y_train3)

#Stress Detection Page
if rad=="Stress Detection":
    st.header("Detect The Amount Of Stress In The Text!!")
    st.text("")
    st.markdown('<div class="custom-text">Determine stress levels from the text\'s tone and content.</div>', unsafe_allow_html=True)
    st.text("")
    st.text("")
    sent3=st.text_area("Enter The Text")
    transformed_sent3=transform_text(sent3)
    vector_sent3=tfidf3.transform([transformed_sent3])
    prediction3=model3.predict(vector_sent3)[0]

    if st.button("Predict"):
        if prediction3>=0:
            st.warning("Stressful Text!!")
        elif prediction3<0:
            st.success("Not A Stressful Text!!")

#Hate & Offensive Content Prediction
tfidf4=TfidfVectorizer(stop_words=sw,max_features=20)
def transform4(txt1):
    txt2=tfidf4.fit_transform(txt1)
    return txt2.toarray()

df4=pd.read_csv("Hate Content Detection.csv")
df4=df4.drop(["Unnamed: 0","count","neither"],axis=1)
df4.columns=["Hate Level","Offensive Level","Class Level","Text"]
x=transform4(df4["Text"])
y=df4["Class Level"]
x_train4,x_test4,y_train4,y_test4=train_test_split(x,y,test_size=0.1,random_state=0)
model4=RandomForestClassifier()
model4.fit(x_train4,y_train4)

#Hate & Offensive Content Page
if rad=="Hate and Offensive Content Detection":
    st.header("Detect The Level Of Hate & Offensive Content In The Text!!")
    st.text("")
    st.markdown('<div class="custom-text">Identify any hateful or offensive content.</div>', unsafe_allow_html=True)
    st.text("")
    st.text("")
    sent4=st.text_area("Enter The Text")
    transformed_sent4=transform_text(sent4)
    vector_sent4=tfidf4.transform([transformed_sent4])
    prediction4=model4.predict(vector_sent4)[0]

    if st.button("Predict"):
        if prediction4==0:
            st.exception("Highly Offensive Text!!")
        elif prediction4==1:
            st.warning("Offensive Text!!")
        elif prediction4==2:
            st.success("Non Offensive Text!!")

#Sarcasm Detection Prediction
tfidf5=TfidfVectorizer(stop_words=sw,max_features=20)
def transform5(txt1):
    txt2=tfidf5.fit_transform(txt1)
    return txt2.toarray()

df5=pd.read_csv("Sarcasm Detection.csv")
df5.columns=["Text","Label"]
x=transform5(df5["Text"])
y=df5["Label"]
x_train5,x_test5,y_train5,y_test5=train_test_split(x,y,test_size=0.1,random_state=0)
model5=LogisticRegression()
model5.fit(x_train5,y_train5) 

#Sarcasm Detection Page
if rad=="Sarcasm Detection":
    st.header("Detect Whether The Text Is Sarcastic Or Not!!")
    st.text("")
    st.markdown('<div class="custom-text">A unique feature that helps detect sarcasm in the text.\n\n\n</div>', unsafe_allow_html=True)
    st.text("")
    st.text("")
    sent5=st.text_area("Enter The Text")
    transformed_sent5=transform_text(sent5)
    vector_sent5=tfidf5.transform([transformed_sent5])
    prediction5=model5.predict(vector_sent5)[0]

    if st.button("Predict"):
        if prediction5==1:
            st.exception("Sarcastic Text!!")
        elif prediction5==0:
            st.success("Non Sarcastic Text!!")


################## ------- STYLING OF THE PAGE -------
#hiding Streamlit default styling
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)