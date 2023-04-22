"""
Spam Classifier API
"""
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

from string import punctuation 

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
):
    nltk.download(dependency)

app = Flask(__name__)

model = pickle.load(open('spam_detector.bin', 'rb'))

nltk.download('stopwords')
nltk.download('omw-1.4')

stop_words =  stopwords.words('english')
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    # removing stop words
    text = text.split()
    text = [w for w in text if not w in stop_words]
    text = " ".join(text)
    # stemming
    text = text.split()
    lemmatizer = WordNetLemmatizer() 
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmatized_words)

    return(text)

@app.route("/")
def Home():
	return jsonify({"status":"online"})

@app.route("/predict", methods = ["POST"])
def Predict():
    req_data = request.get_json(force=True);
    #print(sms_data['age'])
    sms_content = req_data['data']
    sms_content = text_cleaning(sms_content).lower()
    result = model.predict(pd.Series(sms_content))
    print(result)
    if(result[0]==0):
        return jsonify({"result":"not_spam"})
    else:
        return jsonify({"result":"spam"})

@app.route("/predict_all", methods=["POST"])
def PredictAll():
    req_data = request.get_json(force=True)
    print(req_data)
    sms_data_list = [text_cleaning(s["data"]).lower() for s in req_data]
    X = pd.Series(np.array(sms_data_list))
    result = model.predict(X)
    d1 = {"result":"spam"}
    d2 = {"result":"not_spam"}
    resp = [(d2 if s==0 else d1) for s in result]
    print(resp)
    return jsonify(resp)



if __name__ == "__main__":
	app.run(debug=True)
