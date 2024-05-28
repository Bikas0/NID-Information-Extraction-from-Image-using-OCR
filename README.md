# NID OCR Project

This project is designed to extract text from images of National Identification (NID) cards. The code uses the Ultralytics YOLO model for image segmentation and a combination of PyTesseract and EasyOCR for text extraction.

## Features

- Automatic detection of the NID card side (front or back)
- Segmentation of the NID card image to extract relevant regions
- Text extraction using PyTesseract and EasyOCR
- Handling of both accurate and inaccurate OCR results

## Requirements

- Python 3.10
- PyTorch
- Ultralytics YOLO
- EasyOCR
- PyTesseract

You can install the required packages using the following command:

```
pip install -r requirements.txt
```

## Usage

1. Ensure that you have the necessary files and directories:
   - The pre-trained YOLO model file (`best.pt`) should be placed in the `/home/bikas/passport/NID_TRAIN/runs/segment/train2/weights/` directory.
   - The NID card image file (`images (2).jpg`) should be placed in the `/home/bikas/passport/NID_TRAIN/nid_images/` directory.
   - The directory for storing the segmented text images (`/home/bikas/passport/NID_TRAIN/NID_Crop_Img`) should exist.

2. Run the `main()` function in the provided code:

```python
response = main(image, model, segmentedTexts)
print(response)
```

The `main()` function will perform the following steps:

- Detect the side of the NID card (front or back) using the YOLO model.
- Segment the NID card image and extract the relevant regions.
- Extract text from the segmented regions using PyTesseract and EasyOCR.
- Combine the results and return the final response.

The output will be either a dictionary with the extracted text or a dictionary with a "remarks" key indicating that the front side of the NID card should be used.

## Customization

If you need to modify the file paths or the model used, you can update the following variables in the code:

- `image`: the path to the NID card image file
- `segmentedTexts`: the directory where the segmented text images will be stored
- `model`: the path to the pre-trained YOLO model file

Additionally, you can customize the text extraction process by modifying the `check_values()` function, which handles the comparison and selection of the best OCR results.

## Contributions

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to create a new issue or submit a pull request.
