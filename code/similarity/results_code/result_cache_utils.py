import numpy as np
from dataclasses import dataclass
from json import JSONEncoder
import os
import json
import typing
import pandas as pd
from similarity.data_stucture.BrainData import ContactKind, ImageKind, BrainData
from enum import Enum

number_to_contacts_kinds = {
    1: ContactKind.High,
    2: ContactKind.Low
}

PUBLIC_ENUMS = {
    "ContactKind": ContactKind
}

@dataclass
class MaskValues:
    test_mask: np.array
    train_mask: np.array


path_prefix_cache = "./files/cache"
path_prefix_csv = "./files/csv"


def deserialize_number_to_contacts(arr):
    arr = [item for sublist in arr for item in sublist]
    return [number_to_contacts_kinds[contact_type_number] for contact_type_number in arr]

def decode_test_result(dct):
    # if "__enum__" in dct:
    #     name, member = dct["__enum__"].split(".")
    #     return getattr(PUBLIC_ENUMS[name], member)
    if "test_mask" in dct and "train_mask" in dct:
        return MaskValues(test_mask=np.array(dct["test_mask"]),
                          train_mask=np.array(dct["train_mask"]))
    if "features_subject_1" in dct and "features_subject_2" in dct:
        dct['features_subject_1'] = np.array(dct['features_subject_1'])
        dct['features_subject_2'] = np.array(dct['features_subject_2'])
        dct['subject1_xyz_coordinates'] = np.array(dct['subject1_xyz_coordinates'])
        dct['subject2_xyz_coordinates'] = np.array(dct['subject2_xyz_coordinates'])
        dct['transformation_matrices'] = [np.array(trasformation_matrix) for trasformation_matrix in dct['transformation_matrices']]
        try:

            dct["subject1_contacts_types"] = deserialize_number_to_contacts(dct["subject1_contacts_types"])
            dct["subject2_contacts_types"] = deserialize_number_to_contacts(dct["subject2_contacts_types"])
        except:
            a = 3
        return dct
    else:
        return dct



# class EnumEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if type(obj) in PUBLIC_ENUMS.values():
#             return {"__enum__": str(obj)}
#         return json.JSONEncoder.default(self, obj)
#
# def as_enum(d):
#     if "__enum__" in d:
#         name, member = d["__enum__"].split(".")
#         return getattr(PUBLIC_ENUMS[name], member)
#     else:
#         return d
#
#
class TestResultsEncoder(JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, MaskValues):
            return {
                 "test_mask": obj.test_mask,
                 "train_mask": obj.train_mask
            }
        return JSONEncoder.default(self, obj)


def read_data(output_folder_path,
               subject_1_index,
               subject_2_index,
               database_name,
               current_contact_kind,
               image_kinds,
               create_transformation_matrix_name,
               min_number_of_contacts,
               size_of_test_group):


    current_file_path = get_file_path(os.path.join(path_prefix_cache,
                                                   output_folder_path),
                                      database_name,
                                      current_contact_kind,
                                      image_kinds,
                                      subject_1_index,
                                      subject_2_index,
                                      create_transformation_matrix_name,
                                      min_number_of_contacts,
                                      size_of_test_group)

    # with open(current_file_path, "r") as f: # '../../../output/cache/spearman_alignment/no_random_cross_validation/eventrelatednatural/ContactKind.High/Animals_Patterns/orthogonal/4/0_1.json'
    #         return json.load(f, object_hook=decode_test_result)

    while True:
        try:
            with open(current_file_path, "r") as f: # '../../../output/cache/spearman_alignment/no_random_cross_validation/eventrelatednatural/ContactKind.High/Animals_Patterns/orthogonal/4/0_1.json'
                return json.load(f, object_hook=decode_test_result)
        except:
            print("problem reading data: ", current_file_path)
            a = 3

def read_metadata(output_folder_path,
              database_name,
              current_contact_kind,
              image_kinds,
              create_transformation_matrix_name,
              min_number_of_contacts):

    current_file_path = os.path.join(get_folder_path(os.path.join(path_prefix_cache,
                                                     output_folder_path),
                                                     database_name,
                                                     current_contact_kind,
                                                     image_kinds,
                                                     create_transformation_matrix_name,
                                                     min_number_of_contacts,
                                                     0),
                                     "metadata.json")
    while True:
        try:
            with open(current_file_path, "r") as f:
                return json.load(f, object_hook=decode_test_result)
        except:
            a = 3




def write_csv(output_folder_path: str,
                   df: pd.DataFrame,
                   database_name: str,
                   current_contact_kind: typing.Any,
                   image_kinds: typing.Any,
                   min_number_of_contacts):

    current_file_path = os.path.join(get_folder_path_csv(os.path.join(path_prefix_csv,
                                                                  output_folder_path),
                                                     database_name,
                                                     current_contact_kind,
                                                     image_kinds,
                                                     min_number_of_contacts),
                                     database_name + "_used_contacts.csv")



    current_output_folder_path, curent_file_name = os.path.split(current_file_path)

    if not os.path.exists(current_output_folder_path):
        try:
            os.makedirs(current_output_folder_path)
        except FileExistsError:
            pass
        except:
            a = 3
    df.to_csv(current_file_path)

def write_metadata(output_folder_path: str,
                   number_of_indexes: int,
                   database_name: str,
                   current_contact_kind: typing.Any,
                   image_kinds: typing.Any,
                   create_transformation_matrix_name: str,
                   min_number_of_contacts):

    current_file_path = os.path.join(get_folder_path(os.path.join(path_prefix_cache,
                                                                  output_folder_path),
                                                     database_name,
                                                     current_contact_kind,
                                                     image_kinds,
                                                     create_transformation_matrix_name,
                                                     min_number_of_contacts,
                                                     size_of_test_group=0),
                                     "metadata.json")

    result = {
        "number_of_indexes": number_of_indexes
    }


    current_output_folder_path, curent_file_name = os.path.split(current_file_path)

    if not os.path.exists(current_output_folder_path):
        try:
            os.makedirs(current_output_folder_path)
        except FileExistsError:
            pass
        except:
            a = 3

    with open(current_file_path, "w") as outfile:
        # a= json.dumps(result, cls=TestResultsEncoder)
        # b = json.loads(a, object_hook=decode_test_result)
        json.dump(result, outfile, cls=TestResultsEncoder)
        b = 3



def write_date(output_folder_path,
               subject_1_index,
               subject_2_index,
               subject1_xyz_coordinates,
               subject2_xyz_coordinates,
               subject1_contacts_hamesphare,
               subject2_contacts_hamesphare,
               features_subject_1,
               features_subject_2,
               features_subject_1_indexes,
               features_subject_2_indexes,
               transformation_matrices,
               masks,
               database_name,
               current_contact_kind,
               image_kinds,
               create_transformation_matrix_name,
               min_number_of_contacts,
               subject1_contacts_types,
               subject2_contacts_types,
               size_of_test_group):


    current_file_path = get_file_path(os.path.join(path_prefix_cache,
                                                   output_folder_path),
                                      database_name,
                                      current_contact_kind,
                                      image_kinds,
                                      subject_1_index,
                                      subject_2_index,
                                      create_transformation_matrix_name,
                                      min_number_of_contacts,
                                      size_of_test_group)


    result = {
        "features_subject_1": features_subject_1,
        "features_subject_2": features_subject_2,
        "transformation_matrices": transformation_matrices,
        "masks": masks,
        "features_subject_1_indexes": features_subject_1_indexes,
        "features_subject_2_indexes": features_subject_2_indexes,
        "subject1_xyz_coordinates": subject1_xyz_coordinates,
        "subject2_xyz_coordinates": subject2_xyz_coordinates,
        "subject1_contacts_hamesphare":  subject1_contacts_hamesphare,
        "subject2_contacts_hamesphare": subject2_contacts_hamesphare,
         "subject1_contacts_types": subject1_contacts_types,
         "subject2_contacts_types": subject2_contacts_types,
         "test_group_size": size_of_test_group
    }

    current_output_folder_path, curent_file_name = os.path.split(current_file_path)

    if not os.path.exists(current_output_folder_path):
        try:
            os.makedirs(current_output_folder_path)
        except FileExistsError:
            pass
        except:
            a = 3

    with open(current_file_path, "w") as outfile:
        # a= json.dumps(result, cls=TestResultsEncoder)
        # b = json.loads(a, object_hook=decode_test_result)
        json.dump(result, outfile, cls=TestResultsEncoder)
        b = 3


def get_folder_path(folder_path, database_name, current_contact_kind, image_kinds, create_transformation_matrix_name, min_number_of_contacts, size_of_test_group):
    return os.path.join(folder_path,
                        database_name,
                        str(current_contact_kind),
                        "_".join([current_image_kind.name for current_image_kind in image_kinds]),
                        create_transformation_matrix_name,
                        str(min_number_of_contacts),
                        "test_group_" + str(size_of_test_group))


def get_folder_path_csv(folder_path, database_name, current_contact_kind, image_kinds, min_number_of_contacts):
    return os.path.join(folder_path,
                        database_name,
                        str(current_contact_kind),
                        "_".join([current_image_kind.name for current_image_kind in image_kinds]),
                        str(min_number_of_contacts))

def get_file_path(folder_path,
                  database_name,
                  current_contact_kind,
                  image_kinds,
                  subject_1_index,
                  subject_2_index,
                  create_transformation_matrix_name,
                  min_number_of_contacts,
                  size_of_test_group):
    return os.path.join(get_folder_path(folder_path,
                                        database_name,
                                        current_contact_kind,
                                        image_kinds,
                                        create_transformation_matrix_name,
                                        min_number_of_contacts,
                                        size_of_test_group),
                        str(subject_1_index) + "_" + str(subject_2_index) + ".json")
