from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import tensorflow as tf

app = FastAPI()

# CORS configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Loading the data
df_embeddings = pd.read_csv('df_embeddings.csv')
styles_df = pd.read_csv('stylesedit.csv', nrows=5000)
styles_df['image'] = styles_df.apply(lambda row: str(row['id']) + ".jpg", axis=1)

# Load the model and the image shape
model = tf.keras.models.load_model('last_edit.h5')

# Ensure the model has an input shape defined
if not hasattr(model, 'input_shape'):
    model.build((None, 224, 224, 3))  # Replace with your actual input shape if known

image_shape = model.input_shape[1:3]

# To read the image to prepare it for prediction
def read_image(img, img_shape=image_shape):
    img = cv2.resize(img, (img_shape[1], img_shape[0]))  # Ensure resizing to the correct shape
    img = img.astype(np.float32) / 255
    img = np.expand_dims(img, axis=0)
    return img

# Predict and return the similarity between the image and the data
def predict(model, image):
    y_pred = model.predict(image)
    df_sample_image = pd.DataFrame(y_pred)
    sample_similarity = linear_kernel(df_sample_image, df_embeddings)
    return sample_similarity

# Normalizing the list of similar images
def normalize_sim(similarity):
    x_min = similarity.min(axis=1)
    x_max = similarity.max(axis=1)
    norm = (similarity - x_min) / (x_max - x_min)[:, np.newaxis]
    return norm

# Get the recommended images based on the similarity, sort it, and then get the image name
def get_recommendations(similarity, df=styles_df):
    sim_scores = list(enumerate(similarity[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:5]
    cloth_indices = [i[0] for i in sim_scores]
    return df['image'].iloc[cloth_indices]

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Create image folder if it doesn't exist and make the file location
    os.makedirs("images", exist_ok=True)
    file_location = f"images/{file.filename}"

    # Read image and predict the similarity of the images
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = read_image(img)
    pred = normalize_sim(predict(model, img))

    # Return a list of the similar image names
    recommendation = get_recommendations(pred)

    # Save the uploaded image on the computer
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Return the JSON response of the model with an information message and the list with the images
    return JSONResponse(content={"info": f"File uploaded and the prediction is {len(recommendation.to_list())} images",
                                 "images": recommendation.to_list()}, status_code=200)
