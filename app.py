
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from io import BytesIO
# import cv2
# import os
# import base64

# # Load your saved TensorFlow model
# model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_res_model_dir')  # Path to your model
# infer = model.signatures["serving_default"]

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# OUTPUT_DIR = "static/"  # Directory to save localized images
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Function to preprocess the image
# def preprocess_image(image: Image.Image, target_size=(224, 224)):
#     image = image.convert("RGB")  # Ensure 3 channels
#     image = image.resize(target_size)
#     image_array = np.asarray(image) / 255.0  # Normalize
#     return np.expand_dims(image_array, axis=0)


# def overlay_bounding_box(original_image, mask, output_path, max_box_ratio=0.5, min_box_area=500):
#     mask = (mask > 0.5).astype(np.uint8)  # Binary threshold
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     orig_width, orig_height = original_image.size
#     mask_height, mask_width = mask.shape[:2]
#     scale_x = orig_width / mask_width
#     scale_y = orig_height / mask_height

#     localized_image = np.array(original_image)

#     # Ensure the image has 3 color channels
#     if len(localized_image.shape) == 2 or localized_image.shape[2] != 3:
#         localized_image = cv2.cvtColor(localized_image, cv2.COLOR_GRAY2BGR)

#     pneumonia_detected = False

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         box_area = w * h
#         image_area = mask_width * mask_height
#         box_ratio = box_area / image_area

#         # Filter invalid bounding boxes
#         if box_ratio < max_box_ratio and box_area > min_box_area:
#             # Scale bounding box coordinates
#             x = int(x * scale_x)
#             y = int(y * scale_y)
#             w = int(w * scale_x)
#             h = int(h * scale_y)

#             # Draw the refined bounding box in red (BGR format)
#             cv2.rectangle(localized_image, (x, y), (x + w, y + h), (0, 0, 255), 8)  # Red box
#             pneumonia_detected = True

#     # Save the output image
#     if pneumonia_detected:
#         cv2.imwrite(output_path, localized_image)
#     return pneumonia_detected, output_path if pneumonia_detected else None




# # Function to overlay bounding box and save the localized image
   
# # def overlay_bounding_box(original_image, mask, output_path, max_box_ratio=0.5, min_box_area=500):
# #     mask = (mask > 0.5).astype(np.uint8)  # Binary threshold
# #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     orig_width, orig_height = original_image.size
# #     mask_height, mask_width = mask.shape[:2]
# #     scale_x = orig_width / mask_width
# #     scale_y = orig_height / mask_height

# #     localized_image = np.array(original_image)
# #     pneumonia_detected = False

#     # for contour in contours:
#     #     x, y, w, h = cv2.boundingRect(contour)
#     #     box_area = w * h
#     #     image_area = mask_width * mask_height
#     #     box_ratio = box_area / image_area

#     #     # Filter invalid bounding boxes
#     #     if box_ratio < max_box_ratio and box_area > min_box_area:
#     #         # Scale bounding box coordinates
#     #         x = int(x * scale_x)
#     #         y = int(y * scale_y)
#     #         w = int(w * scale_x)
#     #         h = int(h * scale_y)

#     #         # Draw the refined bounding box
#     #         # cv2.rectangle(localized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
#     #         cv2.rectangle(localized_image, (x, y), (x + w, y + h), (0,0,255 ), 2)  # red box
#     #         pneumonia_detected = True

#     # # Save the output image
#     # output_image_path = output_path if pneumonia_detected else None
#     # if pneumonia_detected:
#     #     cv2.imwrite(output_path, cv2.cvtColor(localized_image, cv2.COLOR_RGB2BGR))
#     # return pneumonia_detected, output_image_path

   

#     # for contour in contours:
#     #     x, y, w, h = cv2.boundingRect(contour)
#     #     box_area = w * h
#     #     image_area = mask_width * mask_height
#     #     box_ratio = box_area / image_area

#     #     # Filter invalid bounding boxes
#     #     if box_ratio < max_box_ratio and box_area > min_box_area:
#     #         # Scale bounding box coordinates
#     #         x = int(x * scale_x)
#     #         y = int(y * scale_y)
#     #         w = int(w * scale_x)
#     #         h = int(h * scale_y)

#     #         # Draw the refined bounding box in red (BGR format)
#     #         cv2.rectangle(localized_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box
#     #         pneumonia_detected = True

#     # # Save the output image
#     # output_image_path = output_path if pneumonia_detected else None
#     # if pneumonia_detected:
#     #     cv2.imwrite(output_path, cv2.cvtColor(localized_image, cv2.COLOR_RGB2BGR))
#     # return pneumonia_detected, output_image_path


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded image
#         img_bytes = await file.read()
#         image = Image.open(BytesIO(img_bytes))
#         original_image = image.copy()  # Keep the original image

#         # Preprocess the image
#         preprocessed_image = preprocess_image(image)

#         # Perform inference
#         predictions = infer(tf.convert_to_tensor(preprocessed_image, dtype=tf.float32))
#         prediction_mask = predictions['output_0'].numpy()[0, :, :, 0]  # Extract mask

#         # Prepare output paths
#         output_filename = os.path.join(
#             OUTPUT_DIR, f"{os.path.splitext(file.filename)[0]}_localized.png"
#         )

#         # Overlay bounding box (if any) and check for pneumonia
#         pneumonia_detected, localized_image_path = overlay_bounding_box(
#             original_image, prediction_mask, output_filename
#         )

#         if pneumonia_detected:
#             with open(localized_image_path, "rb") as img_file:
#                 img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
#             result = {
#                 "diagnosis": "Pneumonia",
#                 "localized_image": img_base64,
#             }
#         else:
#             result = {
#                 "diagnosis": "No Pneumonia",
#                 "localized_image": None,
#             }

#         return JSONResponse(content=result)

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})






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

# # Load your saved TensorFlow model
# model = tf.saved_model.load('/home/cr7karki/Documents/projects/samir/conversion/saved_res_model_dir')  # Path to your model


model_path = os.path.join(os.getcwd(), 'saved_res_model_dir')

# Load the model
model = tf.saved_model.load(model_path)


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

def overlay_bounding_box(original_image, mask, output_path, max_box_ratio=0.5, min_box_area=500):
    mask = (mask > 0.5).astype(np.uint8)  # Binary threshold
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    orig_width, orig_height = original_image.size
    mask_height, mask_width = mask.shape[:2]
    scale_x = orig_width / mask_width
    scale_y = orig_height / mask_height

    localized_image = np.array(original_image)

    # Ensure the image has 3 color channels
    if len(localized_image.shape) == 2 or localized_image.shape[2] != 3:
        localized_image = cv2.cvtColor(localized_image, cv2.COLOR_GRAY2BGR)

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

            # Draw the refined bounding box in red (BGR format)
            cv2.rectangle(localized_image, (x, y), (x + w, y + h), (0, 0, 255), 8)  # Red box
            pneumonia_detected = True

    # Save the output image
    if pneumonia_detected:
        cv2.imwrite(output_path, localized_image)
    return pneumonia_detected, output_path if pneumonia_detected else None

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

        # Calculate sigmoid probability
        probabilities = tf.nn.sigmoid(predictions['output_0']).numpy()
        max_probability = np.max(probabilities)  # Extract the maximum probability

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
                "probability": round(float(max_probability), 3),
                "localized_image": img_base64,
            }
        else:
            result = {
                "diagnosis": "No Pneumonia",
                "probability": round(float(max_probability), 3),
                "localized_image": None,
            }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


