import os
import numpy as np
import json
import warnings


class CacheHelper:
    def __init__(self, database_name):
        self.cache_folder = "./cache/" + database_name + "/"
        self.subject_contacts_folder_path = os.path.join(self.cache_folder, "subject_contacts")
        self.images_contacts_contacts_folder_path = os.path.join(self.cache_folder, "image_contacts")
        self.contacts_activations_folder_path = os.path.join(self.cache_folder, "contact_activations")
        self.image_embeddings_folder_path = os.path.join(self.cache_folder, "image_embeddings")
        self.subject_has_both_contacts_file_path = os.path.join(self.cache_folder, "subject_has_both_contacts.json")


    def get_subject_contact_file_name(self, subject_name, contact_kind, images_kinds):
        return subject_name + "_" + str(contact_kind) + "_" + self.get_images_kinds_contacenation(images_kinds) + ".txt"

    def get_image_embeddings_file_name(self, model_name, image_name, contact_kind):
        return model_name + "_" + image_name + "_" + str(contact_kind) + ".txt"

    def does_person_have_high_and_low_contacts_from_cache(self, subject_name):
        try:
            with open(self.subject_has_both_contacts_file_path, "r") as f:
                return json.load(f)[subject_name]
        except:
            return None

    def set_person_have_high_and_low_contacts_from_cache(self, subject_name, value):

        with open(self.subject_has_both_contacts_file_path, "r") as f:
            subjects_json = json.load(f)
            subjects_json[subject_name] = value
        with open(self.subject_has_both_contacts_file_path, "w") as f:
            json.dump(subjects_json, f)

    def get_subject_contacts_feature_cache(self, subject_name, contact_kind, images_kinds):
        try:
            if not os.path.exists(self.subject_contacts_folder_path):
                os.makedirs(self.subject_contacts_folder_path)

            file_path = os.path.join(self.subject_contacts_folder_path,
                                     self.get_subject_contact_file_name(subject_name,
                                                                        contact_kind,
                                                                        images_kinds))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="loadtxt: Empty input file.*")
                # .. your divide-by-zero code ..
                return np.loadtxt(file_path)
        except:
            return None

    def set_subject_contacts_feature_cache(self, subject_name, contact_kind, images_kinds, value):

        if not os.path.exists(self.subject_contacts_folder_path):
            os.makedirs(self.subject_contacts_folder_path)

        file_path = os.path.join(self.subject_contacts_folder_path,
                                 self.get_subject_contact_file_name(subject_name,
                                                                    contact_kind,
                                                                    images_kinds))

        try:
            np.savetxt(file_path, value)
        except:
            a = 4


    def get_image_contacts_feature_cache(self, image_name, contact_kind):
        try:
            if not os.path.exists(self.images_contacts_contacts_folder_path):
                os.makedirs(self.images_contacts_contacts_folder_path)

            file_path = os.path.join(self.images_contacts_contacts_folder_path,
                                     self.get_subject_contact_file_name(image_name, contact_kind))
            return np.loadtxt(file_path)
        except:
            return None

    def set_image_contacts_feature_cache(self, image_name, contact_kind, value):

        if not os.path.exists(self.images_contacts_contacts_folder_path):
            os.makedirs(self.images_contacts_contacts_folder_path)

        file_path = os.path.join(self.images_contacts_contacts_folder_path,
                                 self.get_subject_contact_file_name(image_name, contact_kind))
        np.savetxt(file_path, value)


    def get_contact_activation_cache(self, subject_name, contact_name):
        try:
            if not os.path.exists(self.contacts_activations_folder_path):
                os.makedirs(self.contacts_activations_folder_path)

            file_path = os.path.join(self.contacts_activations_folder_path,
                                     self.get_subject_contact_file_name(subject_name, contact_name))
            return np.loadtxt(file_path)
        except:
            return None

    def set_contact_activation_cache(self, subject_name, contact_name, value):

        if not os.path.exists(self.contacts_activations_folder_path):
            os.makedirs(self.contacts_activations_folder_path)

        file_path = os.path.join(self.contacts_activations_folder_path,
                                 self.get_subject_contact_file_name(subject_name, contact_name))
        np.savetxt(file_path, value)

    def get_image_embeddings(self, model_name, image_name, contact_kind):
        try:
            if not os.path.exists(self.image_embeddings_folder_path):
                os.makedirs(self.image_embeddings_folder_path)

            file_path = os.path.join(self.image_embeddings_folder_path,
                                     self.get_image_embeddings_file_name(model_name, image_name, contact_kind))
            return np.loadtxt(file_path)
        except:
            return None

    def set_image_embeddings(self, model_name, image_name, contact_kind, value):

        if not os.path.exists(self.image_embeddings_folder_path):
            os.makedirs(self.image_embeddings_folder_path)

        file_path = os.path.join(self.image_embeddings_folder_path,
                                 self.get_image_embeddings_file_name(model_name, image_name, contact_kind))
        np.savetxt(file_path, value)


    def get_images_kinds_contacenation(self, images_kinds):
        return "_".join([str(image_kind) for image_kind in  images_kinds])



