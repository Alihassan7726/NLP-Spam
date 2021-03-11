# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:32:16 2020

@author: Ali
"""

from flask import Flask,render_template,url_for,request

import pickle
import string , re 
import nltk
import gzip
from nltk.stem import WordNetLemmatizer as wnl
# load the model from disk
with gzip.open('model_2.pkl', 'rb') as ifp:
    m = pickle.load(ifp)

cv=pickle.load(open('vector_f.pkl','rb'))
def clean_text(text):  
    data = [char for char in text if char not in string.punctuation]
    data = ''.join(data)
    data = str(data)
    words = re.sub(r"(@[A-Za-z0-9]+)|([^A-Za-z0-9 \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", data )  
    words = words.lower()
    final_words =  [wnl().lemmatize(word , pos = "v") for word in words.split()]
    final_words = ' '.join(final_words)
    return(final_words)


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message  = str(request.form['message'])
        data = [message]
        data_clean = clean_text(data)
        data_clean = [data_clean]
        vect = cv.transform(data_clean).toarray()
        my_prediction = m.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
