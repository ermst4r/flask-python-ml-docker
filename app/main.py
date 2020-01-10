from flask import Flask,jsonify

import pickle as p
import urllib.parse
from flask import request
import json
from pprint import pprint
app = Flask(__name__)
bow_transformer = p.load(open('bow_transformer.pickle', 'rb'))
tfidf_transformer = p.load(open('tfidf_transformer.pickle', 'rb'))
model = p.load(open('model.pickle', 'rb'))

@app.route("/")
def hello():
    return jsonify({"message":"Willkomen!"})

@app.route('/api/')
def makecalc():
    search = request.args.get('search')
    message = search
    bow4 = bow_transformer.transform([message])
    tfidf4 = tfidf_transformer.transform(bow4)
    counter = 0
    predicted = model.predict_proba(tfidf4)
    returnstring = 'Search query <b> '+ search  + '</b> <br> <hr>'
    for x in model.classes_:
        proba = round(predicted[0][counter], 2)
        if proba > 0.01:
            returnstring += urllib.parse.unquote(x) + ' Correctness:  <b>' + str(proba) + "% </b> <br>"
        counter += 1
    return returnstring

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=80)
