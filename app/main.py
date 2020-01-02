from flask import Flask,jsonify

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify({"about":"Hello world1"})


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
