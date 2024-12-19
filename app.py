# # # # from fastapi import FastAPI, File, UploadFile, HTTPException
# # # # from fastapi.responses import StreamingResponse
# # # # from PIL import Image, ImageDraw
# # # # import tensorflow as tf
# # # # import numpy as np
# # # # from io import BytesIO

# # # # # Load your saved TensorFlow model
# # # # model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_model_dir')  # Path to your model

# # # # # Get the default signature for inference (usually "serving_default")
# # # # infer = model.signatures["serving_default"]

# # # # # Create FastAPI app
# # # # app = FastAPI()

# # # # # Function to process the X-ray image and make predictions
# # # # def process_image(image: Image.Image):
# # # #     # Convert image to RGB and resize to the model input size (224x224)
# # # #     image = image.convert("RGB")
# # # #     image = image.resize((224, 224))  # Resize to the model input size

# # # #     # Convert image to numpy array and normalize it
# # # #     image_array = np.array(image) / 255.0  # Example of normalization

# # # #     # Add batch dimension and convert to tensor
# # # #     input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
# # # #     input_tensor = input_tensor[None, ...]  # Add batch dimension

# # # #     # Perform prediction using the model's signature
# # # #     predictions = infer(input_tensor)  # Use the correct signature to run inference

# # # #     # Debugging: Print out the keys and prediction
# # # #     print(f"Prediction keys: {predictions.keys()}")
# # # #     print(f"Prediction output: {predictions['output_0']}")

# # # #     return predictions

# # # # # Function to draw bounding box if pneumonia is detected
# # # # def draw_bounding_box(image: Image.Image, prediction: np.ndarray):
# # # #     # Example: If pneumonia is detected (thresholding the model output, adjust as necessary)
# # # #     if np.sum(prediction > 0.5) > 0:  # Assuming prediction > 0.5 indicates pneumonia area
# # # #         draw = ImageDraw.Draw(image)
# # # #         # Example: Draw a bounding box (adjust coordinates based on actual prediction)
# # # #         draw.rectangle([50, 50, 150, 150], outline="red", width=5)
# # # #     return image

# # # # @app.post("/predict")
# # # # async def predict(file: UploadFile = File(...)):
# # # #     try:
# # # #         # Read the uploaded image
# # # #         img_bytes = await file.read()
# # # #         image = Image.open(BytesIO(img_bytes))
        
# # # #         # Process image and make predictions
# # # #         prediction = process_image(image)

# # # #         # Extract the prediction tensor from the output
# # # #         prediction_mask = prediction['output_0'].numpy()  # Get the mask from 'output_0'

# # # #         # Assuming the mask is a 224x224 array, thresholding the mask to detect pneumonia regions
# # # #         prediction_sum = np.sum(prediction_mask > 0.5)  # Sum the number of pixels indicating pneumonia

# # # #         # Post-process and draw bounding box if pneumonia detected
# # # #         if prediction_sum > 0:  # If there are any pixels predicted as 1 (pneumonia), draw the box
# # # #             result_image = draw_bounding_box(image, prediction_mask)
# # # #         else:
# # # #             result_image = image

# # # #         # Save the resulting image to BytesIO to send it back
# # # #         img_io = BytesIO()
# # # #         result_image.save(img_io, format="PNG")
# # # #         img_io.seek(0)

# # # #         # Return the image with bounding box if pneumonia is detected
# # # #         return StreamingResponse(img_io, media_type="image/png")
    
# # # #     except Exception as e:
# # # #         raise HTTPException(status_code=400, detail=f"Error processing the image: {str(e)}")

# # # # # To run the app: uvicorn app:app --reload


# # # from fastapi import FastAPI, File, UploadFile, HTTPException
# # # from fastapi.responses import StreamingResponse
# # # from PIL import Image, ImageDraw, ImageFont
# # # from fastapi.middleware.cors import CORSMiddleware
# # # import tensorflow as tf
# # # import numpy as np
# # # from io import BytesIO

# # # # Load your saved TensorFlow model
# # # model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_res_model_dir')  # Path to your model

# # # # Get the default signature for inference (usually "serving_default")
# # # infer = model.signatures["serving_default"]

# # # # Create FastAPI app
# # # app = FastAPI()


# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=["*"],  # Allow all origins or specify your Angular app URL like "http://localhost:4200"
# # #     allow_credentials=True,
# # #     allow_methods=["*"],  # Allow all HTTP methods
# # #     allow_headers=["*"],  # Allow all headers
# # # )

# # # # Function to process the X-ray image and make predictions
# # # def process_image(image: Image.Image):
# # #     # Convert image to RGB and resize to the model input size (224x224)
# # #     image = image.convert("RGB")
# # #     image = image.resize((224, 224))  # Resize to the model input size

# # #     # Convert image to numpy array and normalize it
# # #     image_array = np.array(image) / 255.0  # Example of normalization

# # #     # Add batch dimension and convert to tensor
# # #     input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
# # #     input_tensor = input_tensor[None, ...]  # Add batch dimension

# # #     # Perform prediction using the model's signature
# # #     predictions = infer(input_tensor)  # Use the correct signature to run inference

# # #     return predictions

# # # # Function to draw bounding box if pneumonia is detected
# # # def draw_bounding_box(image: Image.Image, prediction: np.ndarray, message: str):
# # #     draw = ImageDraw.Draw(image)
# # #     # Draw bounding box (adjust as per model output)
# # #     if np.sum(prediction > 0.5) > 0:  # Assuming prediction > 0.5 indicates pneumonia area
# # #         draw.rectangle([50, 50, 150, 150], outline="red", width=5)
# # #     # Add text message
# # #     font = ImageFont.load_default()  # You can load a custom font if necessary
# # #     draw.text((10, 10), message, fill="red", font=font)
# # #     return image

# # # @app.post("/predict")
# # # async def predict(file: UploadFile = File(...)):
# # #     try:
# # #         # Read the uploaded image
# # #         img_bytes = await file.read()
# # #         image = Image.open(BytesIO(img_bytes))
        
# # #         # Process image and make predictions
# # #         prediction = process_image(image)

# # #         # Extract the prediction tensor from the output
# # #         prediction_mask = prediction['output_0'].numpy()  # Get the mask from 'output_0'

# # #         # Assuming the mask is a 224x224 array, thresholding the mask to detect pneumonia regions
# # #         prediction_sum = np.sum(prediction_mask > 0.5)  # Sum the number of pixels indicating pneumonia

# # #         # Prepare message based on prediction
# # #         if prediction_sum > 0:  # If there are any pixels predicted as 1 (pneumonia), draw the box
# # #             message = "Pneumonia Detected"
# # #             result_image = draw_bounding_box(image, prediction_mask, message)
# # #         else:
# # #             message = "No Pneumonia Detected"
# # #             result_image = draw_bounding_box(image, prediction_mask, message)

# # #         # Save the resulting image to BytesIO to send it back
# # #         img_io = BytesIO()
# # #         result_image.save(img_io, format="PNG")
# # #         img_io.seek(0)

# # #         # Return the image with bounding box and message
# # #         return StreamingResponse(img_io, media_type="image/png")
    
# # #     except Exception as e:
# # #         raise HTTPException(status_code=400, detail=f"Error processing the image: {str(e)}")

# # # To run the app: uvicorn app:app --reload



# # from fastapi import FastAPI, File, UploadFile, HTTPException
# # from fastapi.responses import StreamingResponse
# # from PIL import Image, ImageDraw, ImageFont
# # from fastapi.middleware.cors import CORSMiddleware
# # import tensorflow as tf
# # import numpy as np
# # from io import BytesIO
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# # from skimage import measure

# # # Load your saved TensorFlow model
# # model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_res_model_dir')  # Path to your model

# # # Get the default signature for inference (usually "serving_default")
# # infer = model.signatures["serving_default"]

# # # Create FastAPI app
# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # Allow all origins or specify your Angular app URL like "http://localhost:4200"
# #     allow_credentials=True,
# #     allow_methods=["*"],  # Allow all HTTP methods
# #     allow_headers=["*"],  # Allow all headers
# # )

# # # Function to process the X-ray image and make predictions
# # def process_image(image: Image.Image):
# #     # Convert image to RGB and resize to the model input size (224x224)
# #     image = image.convert("RGB")
# #     image = image.resize((224, 224))  # Resize to the model input size

# #     # Convert image to numpy array and normalize it
# #     image_array = np.array(image) / 255.0  # Example of normalization

# #     # Add batch dimension and convert to tensor
# #     input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
# #     input_tensor = input_tensor[None, ...]  # Add batch dimension

# #     # Perform prediction using the model's signature
# #     predictions = infer(input_tensor)  # Use the correct signature to run inference
# #     return predictions

# # # Function to draw bounding box if pneumonia is detected
# # def draw_bounding_box(image: Image.Image, mask: np.ndarray):
# #     # Convert mask to binary by applying a threshold
# #     binary_mask = mask > 0.5  # Assuming mask values > 0.5 are considered part of pneumonia region

# #     # Print out mask details for debugging
# #     print(f"Mask shape: {mask.shape}")
# #     print(f"Binary mask sum: {np.sum(binary_mask)}")  # Check how many pixels are > 0.5

# #     # Use skimage's regionprops to detect bounding boxes
# #     labeled_mask = measure.label(binary_mask)  # Label the connected regions in the binary mask
# #     regions = measure.regionprops(labeled_mask)

# #     print(f"Number of detected regions: {len(regions)}")  # Print number of regions detected

# #     draw = ImageDraw.Draw(image)
    
# #     # Draw bounding boxes for each detected region
# #     for region in regions:
# #         minr, minc, maxr, maxc = region.bbox  # Get bounding box coordinates (y1, x1, y2, x2)
# #         print(f"Bounding box: {(minc, minr, maxc, maxr)}")  # Debug bounding box coordinates
# #         draw.rectangle([minc, minr, maxc, maxr], outline="red", width=5)  # Draw box

# #     return image

# # # Endpoint for prediction
# # @app.post("/predict")
# # async def predict(file: UploadFile = File(...)):
# #     try:
# #         # Read the uploaded image
# #         img_bytes = await file.read()
# #         image = Image.open(BytesIO(img_bytes))
        
# #         # Process image and make predictions
# #         prediction = process_image(image)

# #         # Extract the prediction mask (assumed to be the first output, adjust as needed)
# #         prediction_mask = prediction['output_0'].numpy()[0, :, :, 0]  # Extract the mask (2D array)

# #         # Print prediction mask for debugging
# #         print(f"Prediction mask (first 10x10 region): {prediction_mask[:10, :10]}")

# #         # Draw the bounding box on the image
# #         result_image = draw_bounding_box(image, prediction_mask)

# #         # Save the resulting image to BytesIO to send it back
# #         img_io = BytesIO()
# #         result_image.save(img_io, format="PNG")
# #         img_io.seek(0)

# #         # Return the image with bounding box and message
# #         return StreamingResponse(img_io, media_type="image/png")
    
# #     except Exception as e:
# #         raise HTTPException(status_code=400, detail=f"Error processing the image: {str(e)}")

# # # To run the app: uvicorn app:app --reload



# # import tensorflow as tf
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# # from PIL import Image
# # from fastapi import FastAPI, File, UploadFile, HTTPException
# # from fastapi.responses import StreamingResponse
# # from io import BytesIO
# # from fastapi.middleware.cors import CORSMiddleware
# # from skimage import measure

# # # Load your saved TensorFlow model
# # model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_res_model_dir')  # Path to your model
# # infer = model.signatures["serving_default"]

# # app = FastAPI()
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # Allow all origins or specify your Angular app URL like "http://localhost:4200"
# #     allow_credentials=True,
# #     allow_methods=["*"],  # Allow all HTTP methods
# #     allow_headers=["*"],  # Allow all headers
# # )

# # # Function to process the image and make predictions
# # def process_image(image: Image.Image):
# #     try:
# #         # Convert image to RGB (ensure 3 channels) and resize to model input size (224x224)
# #         image = image.convert("RGB")
# #         image = image.resize((224, 224))

# #         # Convert image to numpy array and normalize
# #         image_array = np.array(image) / 255.0

# #         # Check the shape of the image
# #         if image_array.shape != (224, 224, 3):
# #             raise ValueError(f"Image shape is not correct. Expected (224, 224, 3), got {image_array.shape}")
        
# #         # Log the image shape
# #         print(f"Image shape after processing: {image_array.shape}")

# #         # Add batch dimension and convert to tensor
# #         input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
# #         input_tensor = input_tensor[None, ...]

# #         # Perform inference
# #         predictions = infer(input_tensor)
# #         prediction_mask = predictions['output_0'].numpy()  # Extract mask

# #         return prediction_mask
    
# #     except Exception as e:
# #         raise ValueError(f"Error processing the image: {str(e)}")

# # # Function to draw bounding boxes on the image
# # def draw_bounding_boxes(image: Image.Image, prediction_mask: np.ndarray):
# #     try:
# #         # Ensure the mask is 2D
# #         if prediction_mask.ndim > 2:
# #             prediction_mask = prediction_mask.squeeze()  # Remove extra dimensions (e.g., from 3D to 2D)

# #         # Log the shape of the mask after squeezing
# #         print(f"Mask shape after squeeze: {prediction_mask.shape}")
        

# #         # Convert prediction mask to binary (thresholding)
# #         binary_mask = prediction_mask > 0.5  # Example threshold

# #         # Get the labeled regions from the mask
# #         regions = measure.regionprops(binary_mask.astype(int))

# #         # Convert image to an array to draw bounding boxes on it
# #         image_array = np.array(image)
        
# #         # Create a matplotlib figure and axis to draw the bounding boxes
# #         fig, ax = plt.subplots(1, figsize=(12, 12))
# #         ax.imshow(image)  # Display the image

# #         # Draw bounding boxes on the image for each detected region
# #         for region_pred in regions:
# #             y1_pred, x1_pred, y2_pred, x2_pred = region_pred.bbox
# #             heightReg_pred = y2_pred - y1_pred
# #             widthReg_pred = x2_pred - x1_pred

# #             # Log bounding box coordinates
# #             print(f"Bounding box coordinates: (x1: {x1_pred}, y1: {y1_pred}, x2: {x2_pred}, y2: {y2_pred})")

# #             # Draw the bounding box using matplotlib's patches
# #             ax.add_patch(patches.Rectangle((x1_pred, y1_pred), widthReg_pred, heightReg_pred,
# #                                            linewidth=2, edgecolor='red', facecolor='none'))

# #         # Return the figure with bounding boxes
# #         img_io = BytesIO()
# #         fig.savefig(img_io, format="PNG")
# #         img_io.seek(0)

# #         return img_io

# #     except Exception as e:
# #         raise ValueError(f"Error drawing bounding boxes: {str(e)}")

# # @app.post("/predict")
# # async def predict(file: UploadFile = File(...)):
# #     try:
# #         # Read the uploaded image
# #         img_bytes = await file.read()
# #         image = Image.open(BytesIO(img_bytes))

# #         # Log the original image size
# #         print(f"Original image size: {image.size}")

# #         # Process the image and make predictions
# #         prediction_mask = process_image(image)

# #         # Draw the bounding boxes on the image
# #         img_io = draw_bounding_boxes(image, prediction_mask)

# #         # Return the image with bounding boxes as a streaming response
# #         return StreamingResponse(img_io, media_type="image/png")
    
# #     except Exception as e:
# #         # Log the error for debugging
# #         print(f"Error: {str(e)}")

# #         # Raise an HTTP error with the detailed message
# #         raise HTTPException(status_code=400, detail=f"Error processing the image: {str(e)}")



# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import StreamingResponse
# from io import BytesIO
# from fastapi.middleware.cors import CORSMiddleware
# from skimage import measure
# import base64  
# from fastapi.responses import JSONResponse

# # Load your saved TensorFlow model
# model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_res_model_dir')  # Path to your model
# infer = model.signatures["serving_default"]

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins or specify your Angular app URL like "http://localhost:4200"
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )

# # Function to process the image and make predictions
# def process_image(image: Image.Image):
#     try:
#         # Convert image to RGB (ensure 3 channels) and resize to model input size (224x224)
#         image = image.convert("RGB")
#         image = image.resize((224, 224))

#         # Convert image to numpy array and normalize
#         image_array = np.array(image) / 255.0

#         # Check the shape of the image
#         if image_array.shape != (224, 224, 3):
#             raise ValueError(f"Image shape is not correct. Expected (224, 224, 3), got {image_array.shape}")
        
#         # Log the image shape
#         print(f"Image shape after processing: {image_array.shape}")

#         # Add batch dimension and convert to tensor
#         input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
#         input_tensor = input_tensor[None, ...]

#         # Perform inference
#         predictions = infer(input_tensor)
#         prediction_mask = predictions['output_0'].numpy()  # Extract mask

#         return prediction_mask
    
#     except Exception as e:
#         raise ValueError(f"Error processing the image: {str(e)}")

# # Function to determine labels based on predictions
# def get_label(prediction_mask):
#     # Example logic: Assuming mask values > 0.5 mean "Pneumonia", otherwise "No Pneumonia"
#     if np.any(prediction_mask > 0.5):
#         return "Pneumonia"
#     else:
#         return "No Pneumonia"

# # Function to draw bounding boxes and labels on the image
# def draw_bounding_boxes(image: Image.Image, prediction_mask: np.ndarray):
#     try:
#         # Ensure the mask is 2D
#         if prediction_mask.ndim > 2:
#             prediction_mask = prediction_mask.squeeze()  # Remove extra dimensions (e.g., from 3D to 2D)

#         # Log the shape of the mask after squeezing
#         print(f"Mask shape after squeeze: {prediction_mask.shape}")

#         # Convert prediction mask to binary (thresholding)
#         binary_mask = prediction_mask > 0.5  # Example threshold

#         # Get the labeled regions from the mask
#         regions = measure.regionprops(binary_mask.astype(int))

#         # Convert image to an array to draw bounding boxes on it
#         image_array = np.array(image)
        
#         # Create a matplotlib figure and axis to draw the bounding boxes
#         fig, ax = plt.subplots(1, figsize=(12, 12))
#         ax.imshow(image)  # Display the image

#         # Get the label for the prediction
#         label = get_label(prediction_mask)

#         # Draw bounding boxes on the image for each detected region
#         for region_pred in regions:
#             y1_pred, x1_pred, y2_pred, x2_pred = region_pred.bbox
#             heightReg_pred = y2_pred - y1_pred
#             widthReg_pred = x2_pred - x1_pred

#             # Log bounding box coordinates
#             print(f"Bounding box coordinates: (x1: {x1_pred}, y1: {y1_pred}, x2: {x2_pred}, y2: {y2_pred})")

#             # Draw the bounding box
#             rect = patches.Rectangle((x1_pred, y1_pred), widthReg_pred, heightReg_pred,
#                                      linewidth=2, edgecolor='red', facecolor='none')
#             ax.add_patch(rect)

#             # Draw the label text near the bounding box
#             ax.text(x1_pred, y1_pred - 10, label, color='white',
#                     fontsize=12, backgroundcolor='red')

#         # Return the figure with bounding boxes
#         img_io = BytesIO()
#         fig.savefig(img_io, format="PNG", bbox_inches='tight')
#         plt.close(fig)
#         img_io.seek(0)

#         return img_io

#     except Exception as e:
#         raise ValueError(f"Error drawing bounding boxes: {str(e)}")


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded image
#         img_bytes = await file.read()
#         image = Image.open(BytesIO(img_bytes))

#         # Log the original image size
#         print(f"Original image size: {image.size}")

#         # Process the image and make predictions
#         prediction_mask = process_image(image)
#         label = get_label(prediction_mask)  # Extract label from prediction mask

#         # Draw the bounding boxes on the image
#         img_io = draw_bounding_boxes(image, prediction_mask)

#         # Encode the image to Base64
#         img_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

#         # Return the label and image as a JSON response
#         response = {
#             "label": label,
#             "image": img_base64  # Base64-encoded image
#         }
#         return JSONResponse(content=response)

#     except Exception as e:
#         # Log the error for debugging
#         print(f"Error: {str(e)}")

#         # Raise an HTTP error with the detailed message
#         raise HTTPException(status_code=400, detail=f"Error processing the image: {str(e)}")








import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import cv2
import os
import base64

# Load your saved TensorFlow model
model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_res_model_dir')  # Path to your model
infer = model.signatures["serving_default"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "static/"  # Directory to save localized images
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to preprocess the image
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize(target_size)
    image_array = np.asarray(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)

# Function to overlay bounding box and save the localized image
def overlay_bounding_box(original_image, mask, output_path, max_box_ratio=0.5, min_box_area=500):
    mask = (mask > 0.5).astype(np.uint8)  # Binary threshold
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    orig_width, orig_height = original_image.size
    mask_height, mask_width = mask.shape[:2]
    scale_x = orig_width / mask_width
    scale_y = orig_height / mask_height

    localized_image = np.array(original_image)
    pneumonia_detected = False

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_area = w * h
        image_area = mask_width * mask_height
        box_ratio = box_area / image_area

        # Filter invalid bounding boxes
        if box_ratio < max_box_ratio and box_area > min_box_area:
            # Scale bounding box coordinates
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            # Draw the refined bounding box
            cv2.rectangle(localized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
            pneumonia_detected = True

    # Save the output image
    output_image_path = output_path if pneumonia_detected else None
    if pneumonia_detected:
        cv2.imwrite(output_path, cv2.cvtColor(localized_image, cv2.COLOR_RGB2BGR))
    return pneumonia_detected, output_image_path

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        img_bytes = await file.read()
        image = Image.open(BytesIO(img_bytes))
        original_image = image.copy()  # Keep the original image

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Perform inference
        predictions = infer(tf.convert_to_tensor(preprocessed_image, dtype=tf.float32))
        prediction_mask = predictions['output_0'].numpy()[0, :, :, 0]  # Extract mask

        # Prepare output paths
        output_filename = os.path.join(
            OUTPUT_DIR, f"{os.path.splitext(file.filename)[0]}_localized.png"
        )

        # Overlay bounding box (if any) and check for pneumonia
        pneumonia_detected, localized_image_path = overlay_bounding_box(
            original_image, prediction_mask, output_filename
        )

        if pneumonia_detected:
            with open(localized_image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            result = {
                "diagnosis": "Pneumonia",
                "localized_image": img_base64,
            }
        else:
            result = {
                "diagnosis": "No Pneumonia",
                "localized_image": None,
            }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
