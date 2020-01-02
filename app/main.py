from flask import Flask,jsonify
import numpy as np
import pickle as p
import json
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify({"about":"Hello world1"})

@app.route('/api/')
def makecalc():
    bow_transformer = p.load(open('bow_transformer.pickle', 'rb'))
    tfidf_transformer = p.load(open('tfidf_transformer.pickle', 'rb'))
    model = p.load(open('model.pickle', 'rb'))
    message = 'erwin'
    bow4 = bow_transformer.transform([message])
    tfidf4 = tfidf_transformer.transform(bow4)
    return jsonify(model.predict_proba(tfidf4))



if __name__ == "__main__":

    app.run(host='0.0.0.0', debug=True, port=80)
