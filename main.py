from segmentRoi import roiSegmentation
from ultralytics import YOLO
from utils import frontNID, easyocr_extractText, pytess_extractText, check_values
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = YOLO('/home/bikas/passport/NID_TRAIN/runs/segment/train2/weights/best.pt')
image = "/home/bikas/passport/NID_TRAIN/nid_images/images (2).jpg"
segmentedTexts = "/home/bikas/passport/NID_TRAIN/NID_Crop_Img"


# print("SIDEl : ", side)


#print(text)
def main(image, model, segmentedTexts):

    def check_side(image, model):
        result = model.predict(source=image, classes=[0,1], conf=0.25, device = device)
        
        for res in result:
            res =  res.boxes.cls.tolist().pop()
            # print(res)
        return int(res)

    roiSegmentation(image, segmentedTexts, model)


    side = check_side(image, model)

    if side == 1:
        py_response = pytess_extractText(segmentedTexts)
        py_response = frontNID(py_response)
        #  print("Py Response:",py_response)

        py_state, py_count = check_values(py_response)
        #  print("Py State, Count:", py_state, py_count)

        if not py_state:
            easy_response = easyocr_extractText(segmentedTexts)
            easy_response = frontNID(easy_response)
            #  print("Easy Response:", easy_response)
            easy_state, easy_count = check_values(py_response)
            #  print("Easy State, Count:", easy_state, easy_count)

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

response = main(image, model, segmentedTexts)
print(response)