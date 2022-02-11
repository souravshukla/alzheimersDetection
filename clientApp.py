from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from werkzeug import Response
from com_in_ineuron_ai_utils.utils import decodeImage
from predict import alzheimer

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

#@cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "dataset\\t\\axial_ad100_20.jpg"
        self.classifier = alzheimer(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')
    
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predictionalzheimer()
    # return jsonify(result)
    return Response(result)

port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=port)
    # app.run(debug=True)
