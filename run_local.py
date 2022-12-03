from flask import Flask, jsonify, request

app = Flask(__name__)

from flair.models import TextClassifier
from flair.data import Sentence
classifier = TextClassifier.load('en-sentiment')

def predict(sentence):
    """ Predict the sentiment of a sentence """
    if sentence == "":
        return 0
    text = Sentence(sentence)
    # stacked_embeddings.embed(text)
    classifier.predict(text)
    value = text.labels[0].to_dict()['value'] 
    return 0 if value == 'NEGATIVE' else 1

def flairPredict(sentence):
  result = predict(sentence)
  return result


@app.route('/sentiment', methods = ['GET'])
def samplefunction():
    required_params = ['text']
    missing_params = [key for key in required_params if key not in request.args.keys()]

    if len(missing_params)==0:
        res  = predict(request.args.get('text', ''))
        data = {
                "score": res,
               }

        return jsonify(data)
    else:
         resp = {
                 "status":"failure",
                 "error" : "missing parameters",
                 "message" : "Provide %s in request" %(missing_params)
                }
         return jsonify(resp)

if __name__ == '__main__':
    port = 8000 #the custom port you want
    app.run(host='0.0.0.0', port=port)