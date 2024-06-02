import scipy.io as sio
import os
import json
from PIL import Image
import pandas as pd
import numpy as np


folder_path = "/Users/user/Personal/Master/Thesis/code_for_submission/code/brain_data/Data_4_IDC_collab"
raw_data_foler_path = os.path.join(folder_path, "segMats_rawHFA_smooth50ms")
images_folder = os.path.join(folder_path, "stimuli (3 task versions collapsed)")
subject_code_to_images_file_path = os.path.join(folder_path, "subject_code_to_images.json")
subject_with_electrode_code_to_images_file_path = os.path.join(folder_path, "subject_with_electrode_code_to_images.json")
subject_to_brain_files_file_path = os.path.join(folder_path, "subject_brain_files.json")
elecSummary_allVisual_anatInfo_path = os.path.join(folder_path, "elecSummary_allVisual_anatInfo.csv")
elecSummary_allVisual = os.path.join(folder_path, "elecSummary_allVisual.csv")


def load_file(file_name):
    try:
        return sio.loadmat(os.path.join(raw_data_foler_path, file_name))
    except FileNotFoundError:
         return sio.loadmat(os.path.join(raw_data_foler_path, file_name + "old"))

def get_files_list(dataset_name):
    files_list = [file_name.lower()[:-4] for file_name in os.listdir(raw_data_foler_path) if file_name.endswith(".mat")]
    files_list = [file_name for file_name in files_list if dataset_name in file_name]
    return files_list

def get_images_list():
    images_list = [file_name.lower()[:-4] for file_name in os.listdir(images_folder) if file_name.endswith(".png") or file_name.endswith(".jpg")]
    return images_list

def get_subject_code_to_images(dataset):
    if os.path.exists(subject_code_to_images_file_path):
        with open(subject_code_to_images_file_path, 'r') as f:
            return json.load(f)
    return None

def get_subject_with_electrode_code_to_images(dataset_name):
    if os.path.exists(subject_with_electrode_code_to_images_file_path + "_dataset_name"):
        with open(subject_with_electrode_code_to_images_file_path, 'r') as f:
            return json.load(f)
    return None

def get_subject_to_brain_files():
    if os.path.exists(subject_to_brain_files_file_path):
        with open(subject_to_brain_files_file_path, 'r') as f:
            return json.load(f)
    return None

def set_subject_code_to_brain_files(subject_to_brain_files):
    with open(subject_to_brain_files_file_path, 'w') as f:
        json.dump(subject_to_brain_files, f)

def set_subject_code_to_images(subject_code_to_images):
    with open(subject_code_to_images_file_path, 'w') as f:
        json.dump(subject_code_to_images, f)


def set_subject_with_electrode_code_to_images(subject_with_electrode_code_to_images):
    with open(subject_with_electrode_code_to_images_file_path, 'w') as f:
        json.dump(subject_with_electrode_code_to_images, f)


def is_brain_file_from_higher_electrode(subject_name, electrode_name):
    if electrode_name.startswith("e"):
        electrode_name = electrode_name[1:]
    files_info = pd.read_csv(elecSummary_allVisual_anatInfo_path)
    row = files_info[(files_info.subjNames.str.lower() == subject_name) &
                     (files_info.elecNums == int(electrode_name))]
    return row.ROI_label.shape[0] == 1 and row.ROI_label.iloc[0] == "VTC"

def is_brain_file_from_face_selective(subject_name, electrode_name):
    if electrode_name.startswith("e"):
        electrode_name = electrode_name[1:]
    files_info = pd.read_csv(elecSummary_allVisual)
    row = files_info[(files_info.subjNames.str.lower() == subject_name) &
                     (files_info.elecNums == int(electrode_name))]
    return row.isFaceSelective.shape[0] == 1 and str(row.isFaceSelective.iloc[0]) == "1"


def get_contact_x_y_z_coordinates(subject_name, electrode_name):
    if electrode_name.startswith("e"):
        electrode_name = electrode_name[1:]
    files_info = pd.read_csv(elecSummary_allVisual_anatInfo_path)
    row = files_info[(files_info.subjNames.str.lower() == subject_name) &
                     (files_info.elecNums == int(electrode_name))]
    # if row.shape == 0:
    #     0, 0, 0
    return row.afni_x.iloc[0], row.afni_y.iloc[0], row.afni_z.iloc[0]

def get_contact_hamesphare(subject_name, electrode_name):
    if electrode_name.startswith("e"):
        electrode_name = electrode_name[1:]
    files_info = pd.read_csv(elecSummary_allVisual_anatInfo_path)
    row = files_info[(files_info.subjNames.str.lower() == subject_name) &
                     (files_info.elecNums == int(electrode_name))]
    # if row.shape == 0:
    try:
    #     0, 0, 0
        ctx = row["aparcaseg_bestLabel"].iloc[0].lower()
        if "lh" in ctx or "left" in ctx:
            return "lh"
        if "rh" in ctx or "right" in ctx:
            return "rh"
        else:
            return "unknown"
    except:
        return "unknown"

def is_brain_file_from_lower_electrode(subject_name, electrode_name):
    try:
        files_info = pd.read_csv(elecSummary_allVisual_anatInfo_path)
        row = files_info[(files_info.subjNames.str.lower() == subject_name) &
                         (files_info.elecNums == int(electrode_name))]
        if row.shape[0] == 0:
            return False
        row_value = row.ROI_label.iloc[0]
        return row_value == "V1" or row_value == "V2"
    except:
        a = 3

def get_image(image_name):
    try:
        return Image.open(os.path.join(images_folder, image_name + ".png"))
    except FileNotFoundError:
        return Image.open(os.path.join(images_folder, image_name + "jpg"))