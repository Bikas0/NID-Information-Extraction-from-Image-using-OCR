# from flask import Flask, request, jsonify
# from segmentRoi import roiSegmentation
# from ultralytics import YOLO
# from utils import frontNID, easyocr_extractText, pytess_extractText, check_values
# import torch
# import os

# app = Flask(__name__)

# # Initialize YOLO model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = YOLO('/home/bikas/passport/NID_TRAIN/runs/segment/train2/weights/best.pt')
# segmentedTexts = "/home/bikas/passport/NID_TRAIN/NID_Crop_Img"
# upload_directory = "/home/bikas/passport/NID_TRAIN/upload_img"  # Specify your upload directory here
# os.makedirs(upload_directory, exist_ok=True)


# @app.route('/nid_ocr', methods=['POST'])
# def process_image():
#     # Get image file from request
#     image_file = request.files['image']

#     # Get the original filename
#     filename = image_file.filename
    
#     # Define the path to save the image
#     image_path = os.path.join(upload_directory, filename)
    
#     # Save image to the defined path
#     image_file.save(image_path)

#     # Perform main processing
#     response = main(image_path, model, segmentedTexts)

#     # Delete the uploaded image after processing
#     os.remove(image_path)

#     return jsonify(response)

# def main(image, model, segmentedTexts):

#     def check_side(image, model):
#         result = model.predict(source=image, classes=[0,1], conf=0.25, device=device)
        
#         for res in result:
#             res =  res.boxes.cls.tolist().pop()
#         return int(res)

#     roiSegmentation(image, segmentedTexts, model)
#     side = check_side(image, model)

#     if side == 1:
#         py_response = pytess_extractText(segmentedTexts)
#         py_response = frontNID(py_response)
#         py_state, py_count = check_values(py_response)

#         if not py_state:
#             easy_response = easyocr_extractText(segmentedTexts)
#             easy_response = frontNID(easy_response)
#             easy_state, easy_count = check_values(easy_response)

#             if easy_state:
#                 response = easy_response
#             else:
#                 if easy_count > py_count:
#                     response = easy_response
#                 elif easy_count == py_count:
#                     response = py_response
#                 else:
#                     response = py_response

#         else:
#             response = py_response

#         return response
    
#     return {"remarks" : "Put NID Front Side"}

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port=5001)

from flask import Flask, request, jsonify, send_from_directory, render_template
from segmentRoi import roiSegmentation
from ultralytics import YOLO
from utils import frontNID, easyocr_extractText, pytess_extractText, check_values
import torch
import os

app = Flask(__name__)

# Initialize YOLO model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('/home/bikas/passport/NID_TRAIN/runs/segment/train2/weights/best.pt')
segmentedTexts = "/home/bikas/passport/NID_TRAIN/NID_Crop_Img"
upload_directory = "/home/bikas/passport/NID_TRAIN/upload_img"  # Specify your upload directory here
os.makedirs(upload_directory, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/nid_ocr', methods=['POST'])
def process_image():
    # Get image file from request
    image_file = request.files['image']

    # Get the original filename
    filename = image_file.filename
    
    # Define the path to save the image
    image_path = os.path.join(upload_directory, filename)
    
    # Save image to the defined path
    image_file.save(image_path)

    # Perform main processing
    response = main(image_path, model, segmentedTexts)

    # Delete the uploaded image after processing
    os.remove(image_path)

    return jsonify(response)

def main(image, model, segmentedTexts):

    def check_side(image, model):
        result = model.predict(source=image, classes=[0,1], conf=0.25, device=device)
        
        for res in result:
            res =  res.boxes.cls.tolist().pop()
        return int(res)

    roiSegmentation(image, segmentedTexts, model)
    side = check_side(image, model)

    if side == 1:
        py_response = pytess_extractText(segmentedTexts)
        py_response = frontNID(py_response)
        py_state, py_count = check_values(py_response)

        if not py_state:
            easy_response = easyocr_extractText(segmentedTexts)
            easy_response = frontNID(easy_response)
            easy_state, easy_count = check_values(easy_response)

            if easy_state:
                response = easy_response
            else:
                if easy_count > py_count:
                    response = easy_response
                elif easy_count == py_count:
                    response = py_response
                else:
                    response = py_response

        else:
            response = py_response

        return response
    
    return {"remarks" : "Put NID Front Side"}

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)

