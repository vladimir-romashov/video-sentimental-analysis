"""
@Author: Vladimir Romashov v.romashov
@Description:
This is a Inference module script that is able to detect emotion using CNN models from camera stream or input video.
It uses preprocessing module to prepare date for detection models. Also it uses Output module to display the results.
@Last modified date: 07-10-2020
"""
import os
import sys

import cv2
import numpy as np
from tensorflow.keras import models

from src.preprocessing.face_extractor import FaceDetector
from src.preprocessing.face_alignment import FaceAlignment
from src.preprocessing.spatial_normalizer import SpatialNormalization
from src.preprocessing.frame_generator import FrameGenerator
from src.preprocessing.normalization import Normalization
from src.output.logger import Logger
from src.output.gui_output import GUIOutput
from openvino.inference_engine import IECore


class CNNPrediction:
    """
    CNN Emotion inference: takes a video file or camera stream and predicts emotions using CNN
    """
    __prediction_conf = None
    __model_blob = None
    __model = None
    __model_type = None
    __output_type = None
    __logger = None
    __ie = None
    __enabled_emotions = None

    def __init__(self, predict_conf, enabled_emotions):
        self.__prediction_conf = predict_conf
        self.__enabled_emotions = enabled_emotions
        self.__frame_generator = FrameGenerator(self.__prediction_conf['frame_per_second'])
        self.__face_detector = FaceDetector()
        self.__face_alignment = FaceAlignment()
        self.__logger = Logger()
        self.__gui_output = GUIOutput()
        self.load_model()

    def load_model(self):
        """
        This function:
        - Loads h5 model and IR model for predict emotion
        """
        if self.__prediction_conf['model_format'].lower() == 'h5':
            self.__model = models.load_model(self.__prediction_conf['model_directory'])
        elif self.__prediction_conf['model_format'].lower() == 'ir':
            self.__model, self.__model_blob = self.__get_optimized_model()
        else:
            raise ValueError("model format is not correct")

    def predict_emotion(self):
        """
        This Function predicts the emotion based on its input type
        """
        if self.__is_video_input():
            self.__predict_emotion_video()
        else:
            self.__predict_emotion_camera_stream()

    def __is_video_input(self):
        """
        This function checks if Inference module is running for prerecorded video
        or Camera
        :return True if video file was provided
        """
        if self.__prediction_conf['input_type'].lower() == 'camera':
            return False
        elif self.__prediction_conf['input_type'].lower() == 'video':
            return True
        raise Exception("input_type should be set to Camera or Video in Configuration")

    def __predict_emotion_video(self):
        video = cv2.VideoCapture(self.__prediction_conf['input_directory'])
        faces = self.__get_data_from_video_file(video)
        for face in faces:
            self.__predict(face)

    def __predict_emotion_camera_stream(self):
        camera_stream = cv2.VideoCapture(0)
        camera_stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.__prediction_conf['camera']['CAP_PROP_FRAME_WIDTH'])
        camera_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__prediction_conf['camera']['CAP_PROP_FRAME_HEIGHT'])
        while True:
            timer = timer + 1
            ret, frame = camera_stream.read()
            if timer % self.__prediction_conf['frame_per_second'] == 0:
                self.__predict(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __predict(self, frame):
        if self.__prediction_conf['model_format'].lower() == 'h5':
            self.__h5_model_caller(frame)
        elif self.__prediction_conf['model_format'].lower() == 'ir':
            self.__optimized_model_caller(frame)
        else:
            raise ValueError("model format is not correct")

    def __h5_model_caller(self, frame):
        if self.__is_video_input():
            self.__h5_model_executer(frame, frame)
        else:
            normalized_face = self.__get_data_from_camera(frame)
            if normalized_face is not None:
                self.__h5_model_executer(frame, normalized_face)

    def __h5_model_executer(self, frame, normalized_face):
        normalized_face_image = self.__image_reshape(normalized_face)
        result = self.__model.predict(normalized_face_image)
        self.display_result(result, frame)

    def __image_reshape(self, image):
        reshaped_image = image.reshape(1, self.__prediction_conf['model_input_shape']['height'],
                                self.__prediction_conf['model_input_shape']['width'],
                                self.__prediction_conf['model_input_shape']['channels'])
        reshaped_image = reshaped_image[:, :, :, 0]
        reshaped_image = np.repeat(np.array(reshaped_image)[..., np.newaxis], 3, -1)
        return reshaped_image

    def __optimized_model_caller(self, frame):
        self.__optimized_model_executer(frame, frame)
        cv2.imshow("Frame", frame)

    def __optimized_model_executer(self, frame, normalized_face):
        prepared_face_image = self.__prepare_image(normalized_face)
        ir_result = self.__model.infer(inputs={self.__model_blob: prepared_face_image})
        self.display_result(ir_result, frame)

    def __prepare_image(self, image):
        n, color, height, width = self.__model.input_info[self.__model_blob].input_data.shape
        images = np.ndarray(shape=(n, color, height, width))
        for i in range(n):
            image = image.transpose((2, 0, 1))
            images[i] = image
        return images

    def __get_data_from_video_file(self, file):
        """
        This function gets a video file and returns preprocessed frames
        :param file
        :return: preprocessed normalized frame with detected face
        """
        video = file
        frames = self.__frame_generator.get_frames(video, self.__prediction_conf['is_rpi'])
        if self.__prediction_conf['preprocessing']['spatial_normalization']:
            faces = self.__spatial_normalizer.get_frames(frames)
        elif self.__prediction_conf['preprocessing']['face_detector']:
            faces = self.__face_detector.get_frames(frames)
            if self.__prediction_conf['preprocessing']['face_alignment']:
                faces = self.__face_alignment.get_frames(frames)
        normalized_faces = self.__normalizer.get_frames(faces)
        return normalized_faces

