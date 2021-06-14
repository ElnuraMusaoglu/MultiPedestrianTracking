from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image
import tensorflow as tf
from os.path import join, dirname, realpath
import cv2
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tracking_multiple.mot_helpers.utils as utils
import tracking_multiple.mot_helpers.visualization_utils as vis_utils
#from tracking_multiple.pedestrian_detection.detection.algorithm_yolov4_tiny import detection_stage_yolov4_tiny
#from tracking_multiple.pedestrian_detection.tools import preprocessing
from tracking_multiple.mot_tracking.kcf_mot import nn_matching
#from tracking_multiple.mot_tracking.kcf_mot.detection import Detection
from tracking_multiple.mot_tracking.kcf_mot.multi_target_tracker import Tracker
#from tracking_multiple.pedestrian_detection.tools import generate_detections as gdet
#from tracking_multiple.mot_helpers import hist_helper
from tracking_multiple.mot_helpers.hist_helper import calc_color_histogram


def tf_no_warning():
    try:
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    except ImportError:
        pass

tf_no_warning()

image_w = 608
image_h = 608


class ModelYOLOv4:
    def __init__(self, sequence_dir, detection_file):
        #opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
        #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=opts))
        self.w = image_w
        self.h = image_h
        self.min_confidence = 30  # 30 percent
        #self.nms_max_overlap = 1.0
        #self.detection_model = detection_stage_yolov4_tiny(self.w, self.h, self.detection_score)
        #self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        self.seq_info = utils.gather_sequence_info(sequence_dir, detection_file)

    def get_detections(self, frame_idx, current_frame):
        #image = Image.fromarray(current_frame[..., ::-1])
        #boxs = self.detection_model.detect_image(image)
        #features = self.encoder(current_frame, boxs)
        #hists = hist_helper.get_hists(current_frame, boxs)
        #detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        #boxes = np.array([d.roi for d in detections])
        #scores = np.array([d.confidence for d in detections])
        #indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        #detections = [detections[i] for i in indices]

        # Load image and generate detections.
        detections = utils.create_detections(self.seq_info["detections"], frame_idx, min_height=10)
        detections = [d for d in detections if d.confidence >= self.min_confidence]

        for detection in detections:
            detection.color_hist = calc_color_histogram(current_frame, detection.roi)

        return detections

class TrackerKCF:
    def __init__(self, success_criteria='mot_challenge', max_cosine_distance=.3, nn_budget=None):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, success_criteria, max_iou_distance=0.7, max_age=5, n_init=2)

    def update(self, frame_id, current_frame, detections):
        try:
            mot_metrics = self.tracker.update_iou(frame_id, current_frame, detections)
            return mot_metrics
        except Exception as ex:
            print('Error in KCF Update: {}'.format(str(ex)))

    def predict(self, frame_id, current_frame):
        try:
            self.tracker.predict(frame_id, current_frame)
        except Exception as ex:
            print('Error in KCF Predict: {}'.format(str(ex)))

    def initiate_for_first_detections(self, current_frame, detections):
        try:
            mot_metrics = self.tracker.initiate_for_first_detections(current_frame, detections)
            return mot_metrics
        except Exception as ex:
            print('Error in KCF initiate_for_first_detections: {}'.format(str(ex)))

    def predict_for_first_detections(self, frame_id, current_frame):
        try:
            mot_metrics = self.tracker.predict_for_first_detections(frame_id, current_frame)
            return mot_metrics
        except Exception as ex:
            print('Error in KCF predict_for_first_detections: {}'.format(str(ex)))


def main(model, tracker, img_dir, result_plot_path, test_gt_path, img_path, show=False):
    frame_list = utils.get_img_list(img_dir)
    frame_list.sort()

    for idx in range(len(frame_list)):
        current_frame = cv2.imread(frame_list[idx])
        #current_frame = cv2.resize((image_w, image_h))
        detections = model.get_detections(idx + 1, current_frame)
        tracker.predict(idx + 1, current_frame)
        mot_metrics = tracker.update(idx + 1, current_frame, detections)
        utils.write_gts(test_gt_path, mot_metrics)
        if show:
           vis_utils.show_tracks(tracker.tracker.tracks, current_frame, idx+1, write_image=True, file_path=img_path)

def run_first_detections_tracking(model, tracker, img_dir, result_plot_path, test_gt_path, img_path):
    frame_list = utils.get_img_list(img_dir)
    frame_list.sort()

    current_frame = cv2.imread(frame_list[0])
    detections = model.get_detections(1, current_frame)
    mot_metrics = tracker.initiate_for_first_detections(current_frame, detections)
    utils.write_gts(test_gt_path, mot_metrics)
    vis_utils.show_tracks(tracker.tracker.tracks, current_frame, 1, write_image=False, file_path=img_path)

    for idx in range(len(frame_list)):
        #if idx is 0:
            #continue
        current_frame = cv2.imread(frame_list[idx])
        mot_metrics = tracker.predict_for_first_detections(idx + 1, current_frame)
        utils.write_gts(test_gt_path, mot_metrics)
        vis_utils.show_tracks(tracker.tracker.tracks, current_frame, idx+1, write_image=False, file_path=img_path)

if __name__ == '__main__':
    data_dir = 'D:/DATASET/2DMOT2015/train/'
    #data_dir = 'D:/DATASET/MOT20/train/'
    result_dir = 'D:/DATASET/RESULT/2DMOT2015/'
    #result_dir = 'D:/DATASET/RESULT/MOT20'
    data_names = sorted(os.listdir(data_dir), reverse=True)
    detection_path = 'D:/DATASET/2DMOT2015_detections/2DMOT2015_POI_train'
    #detection_path = 'D:/DATASET/MOT20_detections/MOT20_POI_train'

    data_names = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
    #data_names = ['ADL-Rundle-8']

    #model_filename = 'D:/PROJECTS/MOT_CNN_DATAASSOCIATION/tracking_multiple/pedestrian_detection/model_data/mars-small128.pb'
    success_criteria = 'single_challenge'  # 'mot_challenge'
    tracking_alg = 'KCF_HOG'

    for data_name in data_names:
        print('Tracking starting for ' + data_name)
        sequence_dir = join(data_dir, data_name)
        detection_file = join(detection_path, data_name) + '.npy'
        img_dir = join(sequence_dir, 'img1')
        result_path = result_dir + '/' + data_name + '/' + tracking_alg
        test_gt_dir = result_path + '/GT'
        test_gt_path = test_gt_dir + '/test.txt'
        result_img_path = join(result_path, 'IMG_IOU')  # IMG_COLOR IMG_IOU
        result_plot_path = join(result_path, 'PLOT')

        if os.path.exists(test_gt_path):
            os.remove(test_gt_path)

        utils.create_dir(test_gt_dir)
        utils.create_dir(result_img_path)
        utils.create_dir(result_plot_path)

        model = ModelYOLOv4(sequence_dir, detection_file)
        tracker = TrackerKCF(success_criteria='mot_challenge')

        main(model, tracker, img_dir, result_plot_path, test_gt_path, result_img_path, show=True)
        #run_first_detections_tracking(model, tracker, img_dir, result_plot_path, test_gt_path, result_img_path)

    print('Tracking completed.')
