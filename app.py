from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from TimeSeriesProject.pipeline.predict import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputText.txt"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")  #dvc repro can be used here
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    text = request.json['text']
    with open(clApp.filename, 'w') as file:
        file.write(text)
    result = clApp.classifier.predict()  # Assuming this method processes the text and provides predictions
    return jsonify(result)

if __name__=="__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080, debug=True)
