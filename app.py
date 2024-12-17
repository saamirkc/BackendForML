# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import StreamingResponse
# from PIL import Image, ImageDraw
# import tensorflow as tf
# import numpy as np
# from io import BytesIO

# # Load your saved TensorFlow model
# model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_model_dir')  # Path to your model

# # Get the default signature for inference (usually "serving_default")
# infer = model.signatures["serving_default"]

# # Create FastAPI app
# app = FastAPI()

# # Function to process the X-ray image and make predictions
# def process_image(image: Image.Image):
#     # Convert image to RGB and resize to the model input size (224x224)
#     image = image.convert("RGB")
#     image = image.resize((224, 224))  # Resize to the model input size

#     # Convert image to numpy array and normalize it
#     image_array = np.array(image) / 255.0  # Example of normalization

#     # Add batch dimension and convert to tensor
#     input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
#     input_tensor = input_tensor[None, ...]  # Add batch dimension

#     # Perform prediction using the model's signature
#     predictions = infer(input_tensor)  # Use the correct signature to run inference

#     # Debugging: Print out the keys and prediction
#     print(f"Prediction keys: {predictions.keys()}")
#     print(f"Prediction output: {predictions['output_0']}")

#     return predictions

# # Function to draw bounding box if pneumonia is detected
# def draw_bounding_box(image: Image.Image, prediction: np.ndarray):
#     # Example: If pneumonia is detected (thresholding the model output, adjust as necessary)
#     if np.sum(prediction > 0.5) > 0:  # Assuming prediction > 0.5 indicates pneumonia area
#         draw = ImageDraw.Draw(image)
#         # Example: Draw a bounding box (adjust coordinates based on actual prediction)
#         draw.rectangle([50, 50, 150, 150], outline="red", width=5)
#     return image

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded image
#         img_bytes = await file.read()
#         image = Image.open(BytesIO(img_bytes))
        
#         # Process image and make predictions
#         prediction = process_image(image)

#         # Extract the prediction tensor from the output
#         prediction_mask = prediction['output_0'].numpy()  # Get the mask from 'output_0'

#         # Assuming the mask is a 224x224 array, thresholding the mask to detect pneumonia regions
#         prediction_sum = np.sum(prediction_mask > 0.5)  # Sum the number of pixels indicating pneumonia

#         # Post-process and draw bounding box if pneumonia detected
#         if prediction_sum > 0:  # If there are any pixels predicted as 1 (pneumonia), draw the box
#             result_image = draw_bounding_box(image, prediction_mask)
#         else:
#             result_image = image

#         # Save the resulting image to BytesIO to send it back
#         img_io = BytesIO()
#         result_image.save(img_io, format="PNG")
#         img_io.seek(0)

#         # Return the image with bounding box if pneumonia is detected
#         return StreamingResponse(img_io, media_type="image/png")
    
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing the image: {str(e)}")

# # To run the app: uvicorn app:app --reload


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np
from io import BytesIO

# Load your saved TensorFlow model
model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_model_dir')  # Path to your model

# Get the default signature for inference (usually "serving_default")
infer = model.signatures["serving_default"]

# Create FastAPI app
app = FastAPI()

# Function to process the X-ray image and make predictions
def process_image(image: Image.Image):
    # Convert image to RGB and resize to the model input size (224x224)
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize to the model input size

    # Convert image to numpy array and normalize it
    image_array = np.array(image) / 255.0  # Example of normalization

    # Add batch dimension and convert to tensor
    input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    input_tensor = input_tensor[None, ...]  # Add batch dimension

    # Perform prediction using the model's signature
    predictions = infer(input_tensor)  # Use the correct signature to run inference

    return predictions

# Function to draw bounding box if pneumonia is detected
def draw_bounding_box(image: Image.Image, prediction: np.ndarray, message: str):
    draw = ImageDraw.Draw(image)
    # Draw bounding box (adjust as per model output)
    if np.sum(prediction > 0.5) > 0:  # Assuming prediction > 0.5 indicates pneumonia area
        draw.rectangle([50, 50, 150, 150], outline="red", width=5)
    # Add text message
    font = ImageFont.load_default()  # You can load a custom font if necessary
    draw.text((10, 10), message, fill="red", font=font)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        img_bytes = await file.read()
        image = Image.open(BytesIO(img_bytes))
        
        # Process image and make predictions
        prediction = process_image(image)

        # Extract the prediction tensor from the output
        prediction_mask = prediction['output_0'].numpy()  # Get the mask from 'output_0'

        # Assuming the mask is a 224x224 array, thresholding the mask to detect pneumonia regions
        prediction_sum = np.sum(prediction_mask > 0.5)  # Sum the number of pixels indicating pneumonia

        # Prepare message based on prediction
        if prediction_sum > 0:  # If there are any pixels predicted as 1 (pneumonia), draw the box
            message = "Pneumonia Detected"
            result_image = draw_bounding_box(image, prediction_mask, message)
        else:
            message = "No Pneumonia Detected"
            result_image = draw_bounding_box(image, prediction_mask, message)

        # Save the resulting image to BytesIO to send it back
        img_io = BytesIO()
        result_image.save(img_io, format="PNG")
        img_io.seek(0)

        # Return the image with bounding box and message
        return StreamingResponse(img_io, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing the image: {str(e)}")

# To run the app: uvicorn app:app --reload
