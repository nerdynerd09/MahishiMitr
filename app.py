from flask import Flask,request
from NDRI_Mobile_Code import firebaseDownload
import os
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello There"

@app.route('/predict')
def predict():
    url = request.args['url']
    result = firebaseDownload(url)
    if result == None:
        return "hehe"
    else:
        return result
        
if __name__=="__main__":
    port = int(os.environ.get('PORT',5000))
    app.run(debug=False,host="0.0.0.0",port=port)