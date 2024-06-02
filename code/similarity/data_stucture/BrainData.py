from utils import brain_set_mat_utils
from enum import Enum
import numpy as np
from cache_helper import CacheHelper
import warnings
warnings.filterwarnings("error")

class AggregationType(Enum):
    NoAggregation = 0
    Avarage = 1,
    Max = 2,
    Min = 3,

class ContactKind(Enum):
    All = 0,
    High = 1,
    Low = 2,
    FaceSelective = 3,
    LowAndFaceSelective = 4,
    NoneAbove = 5

# eventrelatednatural
# class ImageKind(Enum):
#     All = 0,
#     Animals = 1,
#     Patterns = 2,
#     People = 3,
#     Places = 4,
#     Tools = 5,

#Blockdesign
# class ImageKind(Enum):
#     All = 0,
#     Body = 1,
#     Face = 2,
#     House = 3,
#     Patterns = 4,
#     Tool = 5,

#eventrelatedold
# class ImageKind(Enum):
#     # file names ['body12', 'face2', 'face3', 'face4', 'face5', 'face7', 'house1', 'house10', 'house2', 'house3', 'house4', 'house6', 'house7', 'house9', 'pattern1', 'pattern4', 'pattern5', 'tool10', 'tool2', 'tool3', 'tool4', 'tool5', 'tool6']
#     All = 0,
#     Body = 1, # 1 file
#     Face = 2, # 5 files
#     House = 3, # 8 files
#     Pattern = 4,# 3 files
#     Tool = 5, # 6 files

# all databases
class ImageKind(Enum):
    All = 0,
    Animals = 1,
    Body = 2,
    Face = 3,
    House = 4,
    Patterns = 5,
    People = 6,
    Places = 7,
    Tool = 8,
    Tools = 9


class BrainData:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.cache_helper = CacheHelper(dataset_name)
        self.files_list = sorted(self._get_features_files_list())
        self.subects_names = self.get_subject_names(self.files_list)
        self.subects_names_with_electrode = self.get_subjects_with_electrodes(self.files_list)
        self.list_images = sorted(self.get_list_images())
        # self.subject_to_images_list = self.get_subject_code_to_images_dict(self.subects_names,
        #                                                                    self.list_images,
        #                                                                    self.files_list)
        # self.subject_with_electrode = self.get_subject_with_electrode_code_to_images_dict(self.subects_names_with_electrode,
        #                                                                    self.list_images,
        #                                                                    self.files_list)

        self.sorted_subects_names = sorted(list(self.subects_names))
        self.sorted_subects_names_with_electrode = sorted(list(self.subects_names_with_electrode))
        self.subject_to_electrodes = self.get_subject_to_relevant_electrodes()
        self.subject_to_electrode_to_contact_type = self.get_subject_to_relevant_electrodes_to_contact_type()
        a = 3



        # self.subject_to_brain_files = self.get_subject_brain_files_dict(self.subects_names,
        #                                                                    self.files_list)
        #

    def get_subject_names(self, files):
        return sorted(list(set([file_name.split("_")[0] if "imp" not in file_name else file_name.split("_")[0] + "_" + file_name.split("_")[1] for file_name in files])))

    def get_subjects_with_electrodes(self, files_list):
        return set([f"{file_name.split('_')[0]}_{file_name.split('_')[1]}" if "imp" not in file_name else
                    f"{file_name.split('_')[0]}_{file_name.split('_')[1]}_{file_name.split('_')[2]}"
                    for file_name in files_list])

    def get_subject_to_relevant_electrodes(self):
        result = {}
        for subject in self.subects_names:
            relevant_elecrods = [subject_with_electrode for subject_with_electrode in self.sorted_subects_names_with_electrode if subject_with_electrode.startswith(subject)]
            relevant_elecrods = [subject_with_electrode.split("_")[1][1:] if "imp" not in subject_with_electrode else  subject_with_electrode.split("_")[2][1:]  for subject_with_electrode in relevant_elecrods]
            result[subject] = relevant_elecrods

        return result

    def get_subject_to_relevant_electrodes_to_contact_type(self):
        result = {}
        for subject in self.subects_names:
            relevant_elecrods = [subject_with_electrode for subject_with_electrode in self.sorted_subects_names_with_electrode if subject_with_electrode.startswith(subject)]
            relevant_elecrods = [subject_with_electrode.split("_")[1][1:] if "imp" not in subject_with_electrode else  subject_with_electrode.split("_")[2][1:]  for subject_with_electrode in relevant_elecrods]

            for electrode in relevant_elecrods:
                if brain_set_mat_utils.is_brain_file_from_higher_electrode(subject, electrode):
                    result[(subject, electrode)] = ContactKind.High
                elif brain_set_mat_utils.is_brain_file_from_lower_electrode(subject, electrode):
                    result[(subject, electrode)] = ContactKind.Low
                else:
                    result[(subject, electrode)] = ContactKind.NoneAbove

        return result



    def get_file_features(self, file_name, aggregationType=AggregationType.Avarage):
        features = brain_set_mat_utils.load_file(file_name)["segMat"]
        features = features[:, 275:450]
        if aggregationType == AggregationType.Avarage:
            features = np.mean(features)
        if aggregationType == AggregationType.Max:
            features = np.max(features)
        if aggregationType == AggregationType.Min:
            features = np.min(features)

        return features

    def get_files_list(self):
        return self.files_list

    def get_subjects_codes(self):
        return self.sorted_subects_names

    # def get_subjects_with_electrode_codes(self):
    #     return list(self.subject_with_electrode)

    def get_list_of_image_brain_files(self, image_name):
        return [file_name for file_name in self.files_list if file_name.startswith(image_name.lower())]

    def get_list_of_subject_full_name_images(self, image_name):
        return [file_name for file_name in self.files_list if file_name.endswith(image_name.lower() + ".mat")]

    def get_similiar_images_features(self, subject_code1, subject_code2):
        subject1_images, subject2_images = self.get_join_subjects_images(subject_code1, subject_code2)
        subject_1_features = self._get_subject_features(subject_code1, subject1_images)
        subject_2_features = self._get_subject_features(subject_code2, subject2_images)
        return subject_1_features, subject_2_features

    def get_join_subjects_images(self, subject_code1, subject_code2):
        subject1__full_name_imags = self._get_subject_list_brain_files(subject_code1)
        subject2__full_name_imags = self._get_subject_list_brain_files(subject_code2)
        join_files = list(set(self.subject_to_images_list[subject_code1]) &
                          set(self.subject_to_images_list[subject_code2]))

        subject1_images = self._filter_subject_list_images_based_file_names(subject1__full_name_imags, join_files)
        subject2_images = self._filter_subject_list_images_based_file_names(subject2__full_name_imags, join_files)

        return subject1_images, subject2_images

    def _is_image_full_path_in_image_list(self, image_full_path, images_list):
            return any([True if image in image_full_path else False for image in images_list])

    def _filter_subject_list_images_based_file_names(self, subject_images, images_names_list):
        return [image for image in subject_images if self._is_image_full_path_in_image_list(image, images_names_list)]

    def _get_features_files_list(self):
        files_list = brain_set_mat_utils.get_files_list(self.dataset_name)
        # files_list = [file_name.replace("eventrelatedold", "eventrelated") for file_name in files_list]
        return files_list

    def get_list_images(self):
        list_images = brain_set_mat_utils.get_images_list()
        list_images = [image_name for image_name in list_images if self.dataset_name in image_name]
        # list_images = [image_name.replace("_stimuli", "") for image_name in list_images]

        return list_images

    def _get_contact_files(self, subject_name, contact_name):
        return [file_name for file_name in self.files_list if file_name.startswith(subject_name + "_" + contact_name)]
        return files_list

    def _get_subject_list_brain_files(self, subject_code):
        return [file for file in self.files_list if file.startswith(subject_code)]
        # subject_images = self.get_list_of_subject_full_name_images(subject_code)
        # return [image for image in subject_images if self._is_image_full_path_in_image_list(image, self.list_images)]


    def _get_image_list_brain_files(self, image_name):
        image_brain_files = self.get_list_of_image_brain_files(image_name)
        return [image for image in image_brain_files if self._is_image_full_path_in_image_list(image, self.list_images)]

    def _get_subject_with_electrodebrain_files(self, subject_code, electrode_name):
        electrode_images = self.get_list_of_subject_full_name_images("_".join(subject_code, electrode_name))
        return [image for image in electrode_images if self._is_image_full_path_in_image_list(image, self.list_images)]

    def _get_subject_features(self, subject_code, list_images):
        subject_images = self._get_subject_list_brain_files(subject_code)
        list_subject_images = self._filter_subject_list_images_based_file_names(subject_images, list_images)
        return self.get_images_full_name_feature_map(list_subject_images)

    def get_images_full_name_feature_map(self, list_brain_files, image_kind):
        list_brain_files = sorted(list_brain_files)
        feature_map = []
        list_images = self.get_filtered_list_images(image_kind)
        for image_name in list_images:
            current_image_measurements = [brain_file for brain_file in list_brain_files if image_name in brain_file]
            feature_map.append(np.array([self.get_file_features(image, AggregationType.Avarage) for image in current_image_measurements]))
        try:
            return np.array(feature_map)
        except:
            return np.array([features for features in feature_map if features.shape[0] != 0])
            a = 3
        #
        # try:
        #     return np.array(feature_map)
        # except:
        #     a = 3
        # a = 3

    def get_filtered_list_images(self, images_kinds):
        result = []
        for image_kind in images_kinds:
            if image_kind == ImageKind.All:
                return self.list_images

            result += [image for image in self.list_images if image_kind.name.lower() in image]
        return result

    # def get_subject_code_to_images_dict(self, subjects_names, list_images, files_list):
    #     result = brain_set_mat_utils.get_subject_code_to_images()
    #     if result != None:
    #         return result
    #
    #     result = self.create_subject_to_image_dict(subjects_names, list_images, files_list)
    #     brain_set_mat_utils.set_subject_code_to_images(result)
    #
    #     return result

    def get_subject_with_electrode_code_to_images_dict(self, subjects_names, list_images, files_list):
        result = brain_set_mat_utils.get_subject_with_electrode_code_to_images(self.dataset_name)
        if result != None:
            return result

        result = self.create_subject_to_image_dict(subjects_names, list_images, files_list)
        brain_set_mat_utils.set_subject_with_electrode_code_to_images(result)

        return result

    def create_subject_to_image_dict(self, subjects_names, list_images, files_list):
        result = {}
        for subject_name in subjects_names:
            subject_images = []
            for image_name in list_images:
                for file_name in files_list:
                    if subject_name in file_name and image_name in file_name:
                        subject_images.append(image_name)
                        break
            result[subject_name] = subject_images

        return result

    # def get_join_subjects_with_elecrode_images(self, subject_code1, subject_code2):
    #     subject1__full_name_imags = self._get_subject_with_electrode_list_full_name_images(subject_code1)
    #     subject2__full_name_imags = self._get_subject_with_electrode_list_full_name_images(subject_code2)
    #     join_files = list(set(self.subject_with_electrode[subject_code1]) &
    #                       set(self.subject_with_electrode[subject_code2]))
    #
    #     subject1_images = sorted(self._filter_subject_list_images_based_file_names(subject1__full_name_imags, join_files))
    #     subject2_images = sorted(self._filter_subject_list_images_based_file_names(subject2__full_name_imags, join_files))
    #
    #     return subject1_images, subject2_images

    def _get_subject_with_electrode_list_full_name_images(self, subject_code):
        subject_images = self.get_list_of_subject_full_name_images(subject_code)
        return [image for image in subject_images if self._is_image_full_path_in_image_list(image, self.list_images)]

    def get_image(self, image_name):
        return brain_set_mat_utils.get_image(image_name)


    def get_xyz_coordinates_all_subjects(self, contact_kind):
        result = []
        for subject in self.sorted_subects_names:
            subject_x_y_z_coordinates = self.get_subject_xyz_coordinates(subject, contact_kind)
            # if len(subject_x_y_z_coordinates) > 0:
            result.append(subject_x_y_z_coordinates)

        return result


    def get_contacts_hamesphare_all_subjects(self, contact_kind):
        result = []
        for subject in self.sorted_subects_names:
            subject_x_y_z_coordinates = self.get_subject_contacts_hamesphare(subject, contact_kind)
            result.append(subject_x_y_z_coordinates)

        return result

    def get_all_subject_contacts_types(self, contact_kind):
        result = []
        for subject in self.sorted_subects_names:
            subject_contacts_type = self.get_subject_contacts_type(subject, contact_kind)
            # if len(subject_x_y_z_coordinates) > 0:
            result.append(subject_contacts_type)

        return result

    def get_subject_contacts_hamesphare(self, subject_name, contact_kind):
        subject_electrodes = self.subject_to_electrodes[subject_name]
        result = []
        for electrode in subject_electrodes:
            if self._check_contact_kind_condition_valid(subject_name, electrode, contact_kind):
                result.append(brain_set_mat_utils.get_contact_hamesphare(subject_name, electrode))

        return result

    def get_subject_contacts_type(self, subject_name, contact_kind):
        subject_electrodes = self.subject_to_electrodes[subject_name]
        result = []
        for electrode in subject_electrodes:
            if not self._check_contact_kind_condition_valid(subject_name, electrode, contact_kind):
                continue
            if (contact_kind == ContactKind.High or contact_kind == ContactKind.All) and \
                    brain_set_mat_utils.is_brain_file_from_higher_electrode(subject_name, electrode):
                result.append(ContactKind.High)
            elif (contact_kind == ContactKind.Low or contact_kind == ContactKind.All or contact_kind == ContactKind.LowAndFaceSelective) and \
                    brain_set_mat_utils.is_brain_file_from_lower_electrode(subject_name, electrode):
                result.append(ContactKind.Low)
            elif (contact_kind == ContactKind.FaceSelective or contact_kind == ContactKind.LowAndFaceSelective) and \
                    brain_set_mat_utils.is_brain_file_from_face_selective(subject_name, electrode):
                result.append(ContactKind.FaceSelective)
            else:
                result.append(None)

        return result


    def get_subject_xyz_coordinates(self, subject_name, contact_kind):
        subject_electrodes = self.subject_to_electrodes[subject_name]
        result = []
        for electrode in subject_electrodes:
            if self._check_contact_kind_condition_valid(subject_name, electrode, contact_kind):
                result.append(brain_set_mat_utils.get_contact_x_y_z_coordinates(subject_name, electrode))

        return np.array(result)

    def get_subject_coordinates_hamesphare(self, subject_name, contact_kind):
        subject_electrodes = self.subject_to_electrodes[subject_name]
        result = []
        for electrode in subject_electrodes:
            if self._check_contact_kind_condition_valid(subject_name, electrode, contact_kind):
                result.append(brain_set_mat_utils.get_contact_x_y_z_coordinates(subject_name, electrode))

        return np.array(result)


    def _check_contact_kind_condition_valid(self, subject_name, contact_name, contact_kind):
        if contact_kind == ContactKind.NoneAbove:
            return True

        if contact_kind == ContactKind.All and (brain_set_mat_utils.is_brain_file_from_higher_electrode(subject_name, contact_name) or
                                               brain_set_mat_utils.is_brain_file_from_lower_electrode(subject_name, contact_name)):
            return True

        if contact_kind == ContactKind.High and brain_set_mat_utils.is_brain_file_from_higher_electrode(subject_name, contact_name):
            return True

        if contact_kind == ContactKind.Low and brain_set_mat_utils.is_brain_file_from_lower_electrode(subject_name, contact_name):
            return True

        if contact_kind == ContactKind.FaceSelective and brain_set_mat_utils.is_brain_file_from_face_selective(subject_name, contact_name):
            return True

        if contact_kind == ContactKind.LowAndFaceSelective and (brain_set_mat_utils.is_brain_file_from_face_selective(subject_name, contact_name) or
                                                                brain_set_mat_utils.is_brain_file_from_lower_electrode(subject_name, contact_name)):
            return True

        return False

    def get_contacts_activations(self, contact_kind):
        result = []
        i = 0
        for subject_with_electrode in self.sorted_subects_names_with_electrode:
            subject_name, contact_name = self._get_contact_and_subject_names_from_subject_with_elcetrode(subject_with_electrode)
            if not self._check_contact_kind_condition_valid(subject_name, contact_name, contact_kind):
                continue
            contact_activations = self.get_contact_feautres(subject_name, contact_name)
            result.append(contact_activations)
            i += 1
        print("activations ", str(i))
        return np.array(result)

    def get_contact_feautres(self, subject_name, contact_name):
        contact_activation_cache = self.cache_helper.get_contact_activation_cache(subject_name, contact_name)
        if contact_activation_cache is None:
            contact_files = self._get_contact_files(subject_name, contact_name)
            contact_activation_cache = np.array(
                [(self.get_file_features(contact_file)) for contact_file in contact_files])
            self.cache_helper.set_contact_activation_cache(subject_name, contact_name, contact_activation_cache)

        return (contact_activation_cache)


    def get_subjects_gram_matrices(self,
                                   contacts_kind,
                                   use_mean=False,
                                   transform_feature_map=lambda x: x,
                                   make_feature_list=False,
                                   use_people_with_high_and_low_contacts=False):
        if contacts_kind == ContactKind.All:
            return self.get_brain_similarity_gram(use_mean, transform_feature_map, make_feature_list)
        if contacts_kind == ContactKind.High:
            return self.get_brain_level_feature_image(True, use_mean, transform_feature_map, make_feature_list, use_people_with_high_and_low_contacts)

        return self.get_brain_level_feature_image(False, use_mean, transform_feature_map, make_feature_list, use_people_with_high_and_low_contacts)

    def does_subject_have_both_high_and_low_electrodes_type(self, subject):
        subject_brain_files = sorted(self._get_subject_list_brain_files(subject))
        does_subject_have_lower_electrode = False
        does_subject_have_high_electrode = False
        for brain_file in subject_brain_files:
            electrode_name = self.get_electrode_name_from_brain_file_name(brain_file)
            if brain_set_mat_utils.is_brain_file_from_higher_electrode(subject, electrode_name):
                does_subject_have_high_electrode = True
            if brain_set_mat_utils.is_brain_file_from_lower_electrode(subject, electrode_name):
                does_subject_have_lower_electrode = True
            if does_subject_have_high_electrode and does_subject_have_lower_electrode:
                return True
        return False


    def get_brain_level_feature_image(self,
                                      use_mean=False,
                                      transform_feature_map=lambda x: x,
                                      make_feature_list=False,
                                      use_people_with_high_and_low_contacts=False,
                                      contact_type=ContactKind.All,
                                      images_kinds=[ImageKind.All]):
        result = None
        list_result = []
        number_of_subjects_with_relevant_features = 0
        for dddd, subject in enumerate(self.sorted_subects_names):
            if use_people_with_high_and_low_contacts:
                subject_have_both_electrodes_cache = self.cache_helper.does_person_have_high_and_low_contacts_from_cache(subject)
                if subject_have_both_electrodes_cache == None:
                    subject_have_both_electrodes_cache = self.does_subject_have_both_high_and_low_electrodes_type(subject)
                    self.cache_helper.set_person_have_high_and_low_contacts_from_cache(subject, subject_have_both_electrodes_cache)

                if not subject_have_both_electrodes_cache:
                    continue

            cache_value = self.cache_helper.get_subject_contacts_feature_cache(subject, contact_type, images_kinds)

            # cache_value = None
            if cache_value is not None:

                features = cache_value
                if len(features.shape) == 1:
                    if features.shape[0] == 0:
                        features = np.zeros((1, 1))
                    else:
                        features = features.reshape(features.shape[0], -1)
            else:
                relevant_brain_files = []
                all_brain_files = sorted(self._get_subject_list_brain_files(subject))

                subject_brain_files = []
                for image_kind in images_kinds:
                    subject_brain_files += self.filter_subjects_images(all_brain_files, image_kind)

                for brain_file in subject_brain_files:
                    electrode_name = self.get_electrode_name_from_brain_file_name(brain_file)
                    if self._check_contact_kind_condition_valid(subject, electrode_name, contact_type):
                        relevant_brain_files.append(brain_file)

                features = self.get_images_full_name_feature_map(relevant_brain_files, images_kinds)
                self.cache_helper.set_subject_contacts_feature_cache(subject, contact_type, images_kinds, features)
                a = 3


            number_of_subjects_with_relevant_features += 1

            transform_features = transform_feature_map(features)

            if make_feature_list:
                try:
                    if len(transform_features.shape) > 1 and transform_features.shape[1] != 0:
                        list_result.append(transform_features)
                    else:
                        list_result.append(np.array([]))
                except:
                    a = 3
            elif result is None:
                result = transform_features
            else:
                if use_mean:
                    result = result + transform_features
                else:
                    result = np.append(result, transform_features, axis=1)
            # if len(list_result) != 0 or result.shape[1] != 0: # TODO remove
            #     break
        if use_mean:
            result = result / number_of_subjects_with_relevant_features

        return list_result if make_feature_list else result

    def filter_subjects_images(self, subject_brain_files, image_kind):
        if image_kind == ImageKind.All:
            return subject_brain_files
        return [brain_file for brain_file in subject_brain_files if image_kind.name.lower() in brain_file]


    def get_brain_similarity_gram(self, use_mean=True, transform_feature_map=lambda x: x, make_feature_list=False):
        transformed_features = None
        list_result = []
        for subject_name in self.sorted_subects_names:
            subject_list_images = self._get_subject_list_brain_files(subject_name)
            features_subject = self.get_images_full_name_feature_map(subject_list_images)

            current_transformed_features = transform_feature_map(features_subject)
            if transformed_features is None:
                transformed_features = current_transformed_features
                list_result.append(current_transformed_features)
            else:
                if use_mean:
                    transformed_features += current_transformed_features
                else:
                    if make_feature_list:
                        list_result.append(current_transformed_features)
                    else:
                        transformed_features = np.append(transformed_features, current_transformed_features, axis=1)
        if use_mean:
            transformed_features = transformed_features / len(self.subects_names)
        return list_result if make_feature_list else transformed_features


    def get_electrode_name_from_brain_file_name(self, brain_file_name):
        if "imp" in brain_file_name:
            return brain_file_name.split("_")[2][1:]

        return brain_file_name.split("_")[1][1:]

    def get_subject_name_from_brain_file_name(self, brain_file_name):
        return brain_file_name.split("_")[0]

    def _get_contact_and_subject_names_from_subject_with_elcetrode(self, subject_with_electrode):
        splitted_subject_with_electrode = subject_with_electrode.split("_")
        electrode_name = splitted_subject_with_electrode[1]
        if electrode_name.startswith("e"):
            electrode_name = electrode_name[1:]
        return splitted_subject_with_electrode[0], electrode_name






