import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import tensorflow.keras as keras
from collections import namedtuple

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')

# Define model directory
MODEL_DIR = "/opt/ml/model" #/model

# Define global variables
model = None
label_encoders = None
scaler = None
tokenizer = None
titles_df = None

# Define features
categorical_features = ["opportunity_geography_type", "opportunity_reporting_territory", "qli_media_type", "qli_rights", "title_primary_genre"]
numerical_features = ["opportunity_deal_amount", "title_runtime", "title_production_year"]
date_features = ["opp_start_date", "opp_expected_close_date", "qli_start_date", "qli_end_date"]
text_features = ["title_synopsis"]
max_len = 100  # Max length for text sequences

#def ping():
#    """Health check route for SageMaker"""
#    return Response(response="{}", status=200, mimetype="application/json")
    
def model_fn(model_dir):
    """
    Load the model and preprocessing artifacts.
    """
    print(' ---- $$$$ ---- $$$$ --- MODEL FN')
    
    global model, label_encoders, scaler, tokenizer, titles_df
    model_path = os.path.join(model_dir, "1")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = tf.keras.models.load_model(model_path)

    print('Model loaded')    
    return model

# Input function to parse request input_fn(request_body, request_content_type)
def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8').strip()
        print('Raw request data: {d}')
        # Parse JSON
        input_data = json.loads(d)
        print('input_data.instances[0]')
        print(input_data["instances"][0])
        opportunity_data = input_data["instances"][0]
        processed_instances = preprocess_opportunity_data(opportunity_data, titles_df, categorical_features, numerical_features, date_features, text_features, label_encoders, scaler, tokenizer, max_len=100)

        print('preprocess instances: ')
        print(processed_instances)
        return processed_instances # {"instances": processed_instances}
    else:
        raise ValueError("Unsupported content type: {}".format(context.request_content_type))

# Preprocess input data
"""
def preprocess_input(input_data):

    print(' ---- $$$$ ---- $$$$ --- PROCESS INPUT')
    print('input data')
    print(input_data)
    print(type(input_data))
    # Convert input to dataframe
    df = pd.DataFrame([input_data])
    
    # Process categorical features
    for col in categorical_features:
        df[col] = label_encoders[col].transform(df[col])

    # Process numerical features
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Process date features (convert to days since reference date)
    for col in date_features:
        df[col] = pd.to_datetime(df[col])
        df[col] = (df[col] - datetime(1970, 1, 1)).dt.days

    # Process text features (tokenize and pad sequences)
    text_sequences = tokenizer.texts_to_sequences(df[text_features[0]])
    text_sequences = pad_sequences(text_sequences, maxlen=max_len)

    # Return processed input data for prediction
    return [df[categorical_features].values, df[numerical_features + date_features].values, text_sequences]
"""

# Function to preprocess opportunity data
def preprocess_opportunity_data(opportunity_data, titles_df, categorical_features, numerical_features, date_features, text_features, label_encoders, scaler, tokenizer, max_len=100):
    """
    Preprocess opportunity data and combine it with each title's features.
    Handles unseen categorical labels by mapping them to an "unknown" category.
    """
    # Load preprocessing artifacts from the "code" folder
    #label_encoders = joblib.load(os.path.join(MODEL_DIR, "code", "label_encoders.pkl"))
    #scaler = joblib.load(os.path.join(MODEL_DIR, "code", "scaler.pkl"))
    #tokenizer = joblib.load(os.path.join(MODEL_DIR, "code", "tokenizer.pkl"))
    #titles_df = pd.read_csv(os.path.join(MODEL_DIR, "code", "titles.csv"))

    print("Loaded label encoders:", label_encoders.keys()) 

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

    # Convert input to tensors
    input_tensors = {
        "input_cat": tf.convert_to_tensor(X_cat_all, dtype=tf.float32),
        "input_num": tf.convert_to_tensor(X_num_all, dtype=tf.float32),
        "input_text": tf.convert_to_tensor(X_text_all, dtype=tf.float32),
    }
    
    return input_tensors # X_cat_all, X_num_all, X_text_all

# Prediction function
def predict_fn(input_data, model):
    """
    Run inference using the model.
    """
    print(' ---- $$$$ ---- $$$$ --- PREDICT FN ')
    print('input data')
    print(input_data)
    print(type(input_data))

    
    #input_cat, input_num, input_text = input_data
    #[input_cat, input_num, input_text]
    
    # Perform inference
    predictions = model.predict(input_data)

    return predictions

# Output function to format response output_fn
#def output_handler(prediction, response_content_type):
#    response = {"predictions": prediction.tolist()}
#    return json.dumps(response)

def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type

