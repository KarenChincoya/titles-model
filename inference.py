import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
import joblib
import os

# Load pre-trained model and preprocessing objects
model = None
label_encoders = None
scaler = None
tokenizer = None
titles_df = None

#MODEL_PATH = os.path.join("/opt/ml/model", "model.h5")
#MODEL_DIR = "/opt/ml/model/saved_model"
MODEL_DIR = "/opt/ml/model/1"

# Define features
categorical_features = ["opportunity_geography_type", "opportunity_reporting_territory", "qli_media_type", "qli_rights", "title_primary_genre"]
numerical_features = ["opportunity_deal_amount", "title_runtime", "title_production_year"]
date_features = ["opp_start_date", "opp_expected_close_date", "qli_start_date", "qli_end_date"]
text_features = ["title_synopsis"]

def ping():
    """Health check route for SageMaker"""
    return Response(response="{}", status=200, mimetype="application/json")

def load_artifacts():
    """
    Load the pre-trained model and preprocessing objects.
    """
    global model, label_encoders, scaler, tokenizer, titles_df
    
    # Load the model from the '1/' directory
    #model = tf.saved_model.load(MODEL_DIR)
    model = tf.keras.models.load_model(MODEL_DIR)

    #model = tf.keras.models.load_model('model.h5')

    # Specify the path to the model directory (the folder containing saved_model.pb)
    #model_dir = "1/"  # Path to your model directory
    # Load the model from the SavedModel directory
    #model = tf.saved_model.load(model_dir)
    #model = TFSMLayer("1/saved_model.pb", call_endpoint='serving_default')

    #print(model.signatures)
    
    label_encoders = joblib.load('/opt/ml/model/code/label_encoders.pkl')
    scaler = joblib.load("/opt/ml/model/code/scaler.pkl")
    tokenizer = joblib.load('/opt/ml/model/code/tokenizer.pkl')
    
    # Load titles data
    titles_df = pd.read_csv('/opt/ml/model/code/titles.csv')
    

# Function to preprocess opportunity data
def preprocess_opportunity_data(opportunity_data, titles_df, categorical_features, numerical_features, date_features, text_features, label_encoders, scaler, tokenizer, max_len=100):
    """
    Preprocess opportunity data and combine it with each title's features.
    Handles unseen categorical labels by mapping them to an "unknown" category.
    """
    # Preprocess opportunity data
    X_cat = []
    for col in categorical_features:
        le = label_encoders[col]
        # Handle unseen labels by mapping them to "unknown"
        if opportunity_data.get(col, "unknown") in le.classes_:
            X_cat.append(le.transform([opportunity_data.get(col, "unknown")])[0])
        else:
            # Map unseen labels to "unknown"
            X_cat.append(le.transform(["unknown"])[0])
    X_cat = np.array(X_cat).reshape(1, -1)  # Shape: (1, num_categorical_features)

    X_num = np.array([[opportunity_data.get(col, 0) for col in numerical_features]])
    X_num = scaler.transform(X_num)  # Scale numerical features

    for col in date_features:
        date_value = datetime.strptime(opportunity_data.get(col, "1970-01-01"), "%Y-%m-%d")
        days_since_reference = (date_value - datetime(1970, 1, 1)).days
        X_num = np.append(X_num, [[days_since_reference]], axis=1)  # Shape: (1, num_numerical_features + num_date_features)

    X_text = tokenizer.texts_to_sequences([opportunity_data.get(text_features[0], "")])
    X_text = pad_sequences(X_text, maxlen=max_len)  # Shape: (1, max_len)

    # Combine opportunity data with each title's features
    X_cat_all = np.tile(X_cat, (len(titles_df), 1))  # Repeat opportunity data for each title
    X_num_all = np.tile(X_num, (len(titles_df), 1))  # Repeat opportunity data for each title

    X_text_all = []
    for title_synopsis in titles_df['title_synopsis']:
        seq = tokenizer.texts_to_sequences([title_synopsis])
        padded_seq = pad_sequences(seq, maxlen=max_len)
        X_text_all.append(padded_seq[0])
    X_text_all = np.array(X_text_all)  # Shape: (num_titles, max_len)

    return X_cat_all, X_num_all, X_text_all

def predict_fn(input_data, model):
    """
    Make predictions using the pre-trained model.
    """
    global titles_df, label_encoders, scaler, tokenizer
    
    # Preprocess input data
    X_cat_all, X_num_all, X_text_all = preprocess_opportunity_data(
        input_data, titles_df, categorical_features, numerical_features, date_features, text_features, label_encoders, scaler, tokenizer
    )
    
    # Make predictions
    #predictions = model.predict([X_cat_all, X_num_all, X_text_all])
    # Add predictions to titles dataframe
    #titles_df['relevance_score'] = predictions
    # Sort titles by relevance score
    #sorted_titles = titles_df.sort_values(by='relevance_score', ascending=False)
    # Return top 10 titles
    #top_titles = sorted_titles[['title_name', 'relevance_score']].head(20).to_dict(orient='records')
    #return top_titles
    # Use the serving signature for inference
    inference_fn = model.signatures['serving_default']
    
    # Convert input to tensors
    input_tensors = {
        "input_1": tf.convert_to_tensor(X_cat_all, dtype=tf.float32),
        "input_2": tf.convert_to_tensor(X_num_all, dtype=tf.float32),
        "input_3": tf.convert_to_tensor(X_text_all, dtype=tf.float32),
    }
    
    # Run inference
    predictions = inference_fn(**input_tensors)["output_0"].numpy()
    
    # Add predictions to titles dataframe
    titles_df["relevance_score"] = predictions.flatten()
    
    # Sort titles by relevance score
    sorted_titles = titles_df.sort_values(by="relevance_score", ascending=False)
    
    # Return top 20 titles
    top_titles = sorted_titles[["title_name", "relevance_score"]].head(20).to_dict(orient="records")
    return top_titles

def input_fn(request_body, request_content_type):
    """
    Parse the input JSON object.
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, content_type):
    """
    Format the prediction output as JSON.
    """
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def model_fn(model_dir):
    """
    Load the model and preprocessing artifacts.
    """
    load_artifacts()
    return model
