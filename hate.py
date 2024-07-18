import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import re
import nltk
nltk.download('stopwords')
#from nltk.util import pr
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))
import streamlit as st
from PIL import Image
image = Image.open('C:/Users/ayush/OneDrive/Desktop/testing/hate-image.png')
st.image(image, caption='',width=700)

global df
def upload_file():
   global df
   uploaded_file = st.file_uploader("Choose a file")
   if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Upload Successful")
   else:
    st.warning("you need to upload a csv or excel file.")

#upload file
upload_file()

# Function to predict hate speech
def predict_hate_speech(text):
    global df
    test_data = text
    if 'class' in df.columns: 
     df['labels']=df['class'].map({0:"Hate Speech Detected", 1:"Offensive language detected", 2:"NO hate and Offensive language detected"})
    elif 'label' in df.columns:
     df['label'] = df['label'].apply(np.ceil) 
     df['labels']=df['label'].map({0:"Hate Speech Detected", 1:"NO hate and Offensive language detected"})
    df = df[['Text','labels']]
    #print(df.head(20))
    def clean(text):
      text = str(text).lower()
      text = re.sub('\[.*?\]', '',text)
      text = re.sub('https?://\S+|www\.\S', '', text) 
      text = re.sub('<.*?>+', '', text)
      text = re.sub(r'[^\w\s]','',text)
      text = re.sub('\n', '', text)
      text = re.sub('\w*\d\w*', '', text)
      text = [word for word in text.split(' ') if word not in stopword]
      text =" ".join(text)
      text = [stemmer.stem(word) for word in text.split(' ')]
      text = " ".join(text)
      return text
    df["Text"] = df["Text"].apply(clean)
    #print(df.head())
    x = np.array(df["Text"])
    y = np.array(df["labels"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size= 0.33, random_state= 42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    st.write("Model Accuracy")
    st.write(accuracy_score(y_test, y_pred))
    df = cv.transform([test_data]).toarray()
    return (clf.predict(df))

# Set the app title
st.title("Hate Speech Detection")


# Add a text input field
text_input = st.text_input("Enter a text:")

# Add a button to trigger the prediction
if st.button("Predict"):
    if text_input:
        prediction = predict_hate_speech(text_input)
        st.write("Prediction:", prediction)
    else:
        st.write("Please enter a text.")