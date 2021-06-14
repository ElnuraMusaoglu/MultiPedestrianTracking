import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list

def get_single_ground_truthes(gt_path):
    gts = {}  # frame, id, x, y, w, h
    with open(gt_path, 'r') as f:
        while True:
            line = f.readline()
            if ',' in line:
                gt_pos = line.split(',')
            else:
                gt_pos = line.split()
            if len(gt_pos) < 4:
                break
            gt_pos_int = [(float(element)) for element in gt_pos[2:6]]
            if not gt_pos[1] in gts:
                gts[gt_pos[1]] = []
            gts[gt_pos[1]].append(gt_pos_int)
    return gts


def write_gts(gt_path, gt_list):
    if not os.path.exists(gt_path):
        file = open(gt_path, "w")
        file.close()
    with open(gt_path, mode='at', encoding='utf-8') as myfile:
        for lines in gt_list:
            myfile.write(','.join(str(line) for line in lines))
            myfile.write('\n')

def create_dir(dir):
    import errno
    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else:
            raise ValueError("Failed to created output directory '%s'" % dir)

def gaussian2d_labels(sz, sigma):
    w, h = sz
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    center_x, center_y = w / 2, h / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5*dist)

    return labels

def gaussian2d_rolled_labels(sz, sigma):
    w, h = sz
    xs, ys = np.meshgrid(np.arange(w)-w//2, np.arange(h)-h//2)
    dist = (xs**2+ys**2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    labels = np.roll(labels, -int(np.floor(sz[0] / 2)), axis=1)
    labels = np.roll(labels, -int(np.floor(sz[1]/2)), axis=0)

    return labels

def calAUC(value_list):
    length = len(value_list)
    delta = 1./(length-1)
    area = 0.
    for i in range(1, length):
        area += (delta*((value_list[i]+value_list[i-1])/2))
    return area

def cos_window(sz):
    """
    width, height = sz
    j = np.arange(0, width)
    i = np.arange(0, height)
    J, I = np.meshgrid(j, i)
    cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
    """

    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])

    return cos_window

def plot_success(gts, preds, save_path, threshold=20):
    threshes, successes = get_thresh_success_pair(gts, preds, threshold)
    plt.plot(threshes, successes, label=str(calAUC(successes))[:5])
    plt.title('Success Plot')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def plot_precision(gts, preds, save_path, threshold=20):
    # x,y,w,h
    threshes, precisions = get_thresh_precision_pair(gts, preds, threshold)
    idx20 = [i for i, x in enumerate(threshes) if x == 20][0]
    plt.plot(threshes, precisions, label=str(precisions[idx20])[:5])
    plt.title('Precision Plots')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def get_thresh_precision_pair(gts, preds, threshold):
    length = min(len(gts), len(preds))
    gts = gts[:length, :]
    preds = preds[:length, :]
    gt_centers_x = (gts[:, 0]+gts[:, 2]/2)
    gt_centers_y = (gts[:, 1]+gts[:, 3]/2)
    preds_centers_x = (preds[:, 0]+preds[:, 2]/2)
    preds_centers_y = (preds[:, 1]+preds[:, 3]/2)
    dists = np.sqrt((gt_centers_x - preds_centers_x) ** 2 + (gt_centers_y - preds_centers_y) ** 2)
    ###
    dists = np.where(dists > 20, dists - 20, dists)
    ###
    threshes = []
    precisions = []
    for thresh in np.linspace(0, 50, 101):
        true_len = len(np.where(dists < thresh)[0])
        precision = true_len / len(dists)
        threshes.append(thresh)
        precisions.append(precision)
    return threshes, precisions

def get_thresh_success_pair(gts, preds, threshold):
    length = min(len(gts), len(preds))
    gts = gts[:length, :]
    preds = preds[:length, :]
    intersect_tl_x = np.max((gts[:, 0], preds[:, 0]), axis=0)
    intersect_tl_y = np.max((gts[:, 1], preds[:, 1]), axis=0)
    intersect_br_x = np.min((gts[:, 0] + gts[:, 2], preds[:, 0] + preds[:, 2]), axis=0)
    intersect_br_y = np.min((gts[:, 1] + gts[:, 3], preds[:, 1] + preds[:, 3]), axis=0)
    intersect_w = intersect_br_x - intersect_tl_x
    intersect_w[intersect_w < 0] = 0
    intersect_h = intersect_br_y - intersect_tl_y
    intersect_h[intersect_h < 0] = 0
    intersect_areas = intersect_h * intersect_w
    ious = intersect_areas / (gts[:, 2] * gts[:, 3] + preds[:, 2] * preds[:, 3] - intersect_areas)
    threshes = []
    successes = []
    for thresh in np.linspace(0, 1, 101):
        success_len = len(np.where(ious > thresh)[0])
        success = success_len / len(ious)
        threshes.append(thresh)
        successes.append(success)
    return threshes, successes

def calc_color_histogram(patch, num_bins=32):
    mask = None

    blue_model = cv2.calcHist([patch], [0], mask, [num_bins],  [0,256]).flatten()
    green_model = cv2.calcHist([patch], [1], mask, [num_bins],  [0,256]).flatten()
    red_model = cv2.calcHist([patch], [2], mask, [num_bins],  [0,256]).flatten()

    color_patch = np.concatenate((blue_model, green_model, red_model))
    color_patch = color_patch/np.sum(color_patch)
    return color_patch

'''
Calculates the Kullback-Lieber divergence
        according to the discrete definition:
        sum [P(i)*log[P(i)/Q(i)]]
        where P(i) and Q(i) are discrete probability
        distributions. In this case the one """

        """ Epsilon is used here to avoid conditional code for
        checking that neither P or Q is equal to 0. """
'''
def discrete_kl_divergence(P, Q):
    epsilon = 0.00001

    # To avoid changing the color model, a copy is made
    temp_P = P + epsilon
    temp_Q = Q + epsilon

    divergence = np.sum(temp_P * np.log(temp_P / temp_Q))
    return divergence

def gather_sequence_info(sequence_dir, detection_file):
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

def create_detections(detection_mat, frame_idx, min_height=0):
    from tracking_multiple.mot_tracking.kcf_mot.detection import Detection
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list