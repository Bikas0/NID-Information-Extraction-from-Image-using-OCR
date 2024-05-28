import re
import easyocr
import os
from natsort import natsorted
from langdetect import detect
import cv2
import pytesseract
from langdetect.lang_detect_exception import LangDetectException
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###   PyTesseract Setup   ###
tess_path = "/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tess_path
os.environ["PATH"] += os.pathsep + os.path.dirname(tess_path)
# os.environ["TESSDATA_PREFIX"] = "/home/bikas/passport/NID_TRAIN/tessdata"
os.environ["TESSDATA_PREFIX"] = "tessdata"

def remove_special_characters(text):
    # The regex pattern [^\w\s] matches any character that is not a word character or whitespace
    cleaned_text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
    return cleaned_text

def delete_files_in_folder(folder_path):
    #print(folder_path)
    try:
        files = os.listdir(folder_path)
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def remove_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

def easyocr_extractText(images_path):
    # Set up the EasyOCR reader
    reader = easyocr.Reader(['bn', 'en'])
    # Loop through each file in the folder
    info = []
    for filename in natsorted(os.listdir(images_path)):
        # Check if the file is an image
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Construct the full file path
            file_path = os.path.join(images_path, filename)

            # Perform text extraction using EasyOCR
            result = reader.readtext(file_path).to(device)

            # Print the extracted text for the current image
            #print(f"Results for {filename}:")
            for detection in result:
                info.append(detection[1])
                #print(detection[1])

    # info = set(info)
    #info.append("15 May 2001")
    #print(info)
    info = remove_duplicates(info)

    delete_files_in_folder(images_path)
    return info


def pytess_extractText(images_path):

    info = []
    for filename in natsorted(os.listdir(images_path)):
        # Check if the file is an image
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Construct the full file path
            file_path = os.path.join(images_path, filename)

            img = cv2.imread(file_path)
            # print(img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # text = pytesseract.image_to_string(gray, lang='ben+eng',config='--psm 8 ')
            text = pytesseract.image_to_string(gray,lang='Bengali',config='--psm 8 ')
            text = text.split("\n")[0]
            text = remove_special_characters(text)
            info.append(text)
            # print(text)

    info = remove_duplicates(info)

    delete_files_in_folder(images_path)
    return info


def frontNID(data):
    def is_bengali(s):
        try:
            return detect(s) == 'bn'
        except LangDetectException:
            return False

    #data = ['রাম্মু চৌণার', 'EAILI CHCINLHLIRY', '14] 1*|', ';7213', 'ঘোমচফHাী [শs']

    # Separate Bangla text and other text
    bangla_text = [item for item in data if is_bengali(item)]
    #print("Bangla Text:", bangla_text)
    other_text = [item for item in data if not is_bengali(item)]
    #print("Other Text:", other_text)

    # Initialize info_dict
    info_dict = {'b_name': '', 'e_name': '', 'f_name': '', 'm_name': '', 'dob': '', 'nid': ''}

    # Process other text
    for item in other_text:
        if info_dict['e_name'].lower().strip() == '' and item.replace(' ', '').isalpha() and not is_bengali(item):
            info_dict['e_name'] = item
        elif info_dict['dob'] == '' and any(char.isdigit() for char in item) and any(char.isalpha() for char in item):
            info_dict['dob'] = item
        elif info_dict['nid'] == '' and all(char.isdigit() or char.isspace() for char in item.strip()):
            info_dict['nid'] = item.strip()
            #info_dict['nid'] = item

    # Process Bangla text
    if len(bangla_text) >= 1:
        # Check if the first element of bangla_text is the same as the first element of data
        if len(bangla_text) < 3 and bangla_text[0] != data[0]:
            # If not, consider it as father's name
            info_dict['f_name'] = bangla_text[0]
            if len(bangla_text) >= 2:
                info_dict['m_name'] = bangla_text[1]
        else:
            # Otherwise, consider it as name
            info_dict['b_name'] = bangla_text[0]
            if len(bangla_text) >= 2:
                info_dict['f_name'] = bangla_text[1]
            if len(bangla_text) >= 3:
                info_dict['m_name'] = bangla_text[2]

    #print("Info Dictionary:", info_dict)

    return info_dict

def check_values(info_dict):
    required_keys = ['b_name', 'e_name', 'f_name', 'm_name', 'dob', 'nid']
    active_count = 0
    all_non_empty = True

    for key in required_keys:
        value = info_dict.get(key, "").strip()
        if value:
            active_count += 1
        else:
            all_non_empty = False
    
    return all_non_empty, active_count





