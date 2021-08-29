# import Flask and jsonify
from flask import Flask, jsonify, request
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

app = Flask(__name__)
api = Api(app)

# load pickle model
model = pickle.load(open("loan_prediction.pickle", "rb" ))

# create an endpoint where we can communicate with model
class Approval(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data)
        # df = pd.read_json(json_data)  
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = model.predict(df)
        # we cannot send numpy array as a result
        return res.tolist()

# assign endpoint
api.add_resource(Approval, '/approval')

if __name__ == '__main__':
    app.run(debug=True)