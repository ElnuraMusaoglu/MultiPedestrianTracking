import numpy as np
import motmetrics as mm
from sklearn.utils.linear_assignment_ import linear_assignment


motchallenge_metric_names = {
    'recall' : 'Rcll',
    'precision' : 'Prcn',
    'num_unique_objects' : 'GT',
    'mostly_tracked' : 'MT',
    'partially_tracked' : 'PT',
    'mostly_lost': 'ML',
    'num_false_positives' : 'FP',
    'num_misses' : 'FN',
    'num_switches' : 'IDs',
    'num_fragmentations' : 'FM',
    'mota' : 'MOTA',
    'motp' : 'MOTP'
}

motchallenge_metric_names2 = {
    'idf1': 'IDF1',
    'idp': 'IDP',
    'idr': 'IDR',
    'recall': 'Rcll',
    'precision': 'Prcn',
    'num_unique_objects': 'GT',
    'mostly_tracked': 'MT',
    'partially_tracked': 'PT',
    'mostly_lost': 'ML',
    'num_false_positives': 'FP',
    'num_misses': 'FN',
    'num_switches': 'IDs',
    'num_fragmentations': 'FM',
    'mota': 'MOTA',
    'motp': 'MOTP',
    'num_transfer': 'IDt',
    'num_ascend': 'IDa',
    'num_migrate': 'IDm',
}

def compute_motchallenge(gt_path, gt_test_path):
    df_gt = mm.io.loadtxt(gt_path)
    df_test = mm.io.loadtxt(gt_test_path)
    return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)

def compute_pairs2(gt_path, gt_test_path):
    matches = {}
    acc = compute_motchallenge(gt_path, gt_test_path)
    matched = acc.mot_events.values[(acc.mot_events.values[:, 0] == 'MATCH')]
    matches_arr = np.array(matched[:, 1:4], dtype='float32')
    matches_arr = matches_arr.round(decimals=1)
    unique_matches = np.unique(matches_arr, axis=0)
    for idx in range(len(unique_matches)):
        if not unique_matches[idx][0] in matches:
            matches[unique_matches[idx][0]] = [unique_matches[idx][1], unique_matches[idx][2]]
        else:
            if unique_matches[idx][2] > matches[unique_matches[idx][0]][1]:
                matches[unique_matches[idx][0]] = [unique_matches[idx][1], unique_matches[idx][2]]

    return matches

def compute_pairs(gt_path, gt_test_path):
    matches = {}
    acc = compute_motchallenge(gt_path, gt_test_path)
    matched = acc.mot_events.values[(acc.mot_events.values[:, 0] == 'MATCH')]
    matches_arr = np.array(matched[:, 1:4], dtype='float32')
    matches_arr = matches_arr.round(decimals=1)
    unique_matches = np.unique(matches_arr, axis=0)
    for idx in range(len(unique_matches)):
        if not unique_matches[idx][0] in matches:
            matches[unique_matches[idx][0]] = unique_matches[idx][1]
    return matches

def get_gt_of_first_detects(gt_path, gt_test_path):
    gts_with_track_id = read_first_frame_gt(gt_path)
    gts_with_track_id_test = read_first_frame_gt(gt_test_path)

    cost_matrix = iou_cost(gts_with_track_id_test, gts_with_track_id)

    matches = match(cost_matrix)

    gts = read_specific_ids(gt_path, matches[:, 1])
    gts_test = read_specific_ids(gt_test_path, matches[:, 0])

    return matches, gts, gts_test

def match(cost_matrix):
    matches = linear_assignment(cost_matrix)
    matches = matches + 1

    return matches

def iou_cost(gt1, gt2):
    cost_matrix = np.zeros((len(gt1), len(gt2)))
    for key in gt1:
        bbox = np.array([gt1[key][0][1], gt1[key][0][2], gt1[key][0][3], gt1[key][0][4]])
        crdn = np.array([np.array(x[0][1:5])for x in list(gt2.values())])
        candidates = np.asarray(crdn)
        cost_matrix[int(key)-1, :] = 1. - iou(bbox, candidates)
    return cost_matrix

def iou(bbox, candidates):
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def read_first_frame_gt(gt_path):
    gts_with_track_id = {}  # frame, id, x, y, w, h
    with open(gt_path, 'r') as f:
        while True:
            line = f.readline()
            if ',' in line:
                gt_pos = line.split(',')
            else:
                gt_pos = line.split()
            if len(gt_pos) < 4 or int(gt_pos[0]) > 1:
                break
            gt_pos_int = [(float(element)) for element in gt_pos[1:6]]
            if not gt_pos[1] in gts_with_track_id:
                gts_with_track_id[gt_pos[1]] = []
            gts_with_track_id[gt_pos[1]].append(gt_pos_int)
    return gts_with_track_id

def read_specific_ids(gt_path, ids):
    gts_with_track_ids = {}  # id, x, y, w, h
    with open(gt_path, 'r') as f:
        while True:
            line = f.readline()
            if ',' in line:
                gt_pos = line.split(',')
            else:
                gt_pos = line.split()
            if len(gt_pos) < 4:
                break
            gt_pos_int = [(float(element)) for element in gt_pos[2:6] if int(gt_pos[1]) in ids]
            if len(gt_pos_int) == 4:
                if not gt_pos[1] in gts_with_track_ids:
                    gts_with_track_ids[gt_pos[1]] = []
                gts_with_track_ids[gt_pos[1]].append(gt_pos_int)

    return gts_with_track_ids