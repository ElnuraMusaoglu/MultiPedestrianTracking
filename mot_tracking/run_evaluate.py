import numpy as np
import os
import motmetrics as mm

import tracking_multiple.mot_helpers.mot_challenge_utils as mot_utils
import tracking_multiple.mot_helpers.utils as single_utils

motchallenge_metrics = [
    'mota',
    'motp',
    'mostly_tracked',
    'mostly_lost',
    'num_switches',
    'num_fragmentations',
    'num_false_positives',
    'num_misses'
]
motchallenge_metrics2 = [
    'mota',
    'motp',
    'idf1',
    'idp',
    'idr',
    'recall',
    'precision',
    'num_unique_objects',
    'mostly_tracked',
    'partially_tracked',
    'mostly_lost',
    'num_false_positives',
    'num_misses',
    'num_switches',
    'num_fragmentations',
    'num_transfer',
    'num_ascend',
    'num_migrate',
]
def metrics_motchallenge_files(dnames, gt_paths, gt_test_paths):
    accs = []
    for i in range(len(dnames)):
        acc = mot_utils.compute_motchallenge(gt_paths[i], gt_test_paths[i])
        accs.append(acc)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=motchallenge_metrics, names=dnames, generate_overall=True)

    print()
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))


def metrics_single_for_first_detections(dnames, gt_paths, gt_test_paths, plot_paths):
    for idx in range(len(dnames)):
        matches, gts, gts_test = mot_utils.get_gt_of_first_detects(gt_paths[idx], gt_test_paths[idx])
        for match in matches:
            real_gt = np.asarray(gts[str(match[1])])
            test_gt = np.asarray(gts_test[str(match[0])])
            plot_eval(str(match[0]), real_gt, test_gt, plot_paths[idx])
            print()
    return


def plot_eval(name, gts, gts_test, plot_path):
    precision_path = os.path.join(plot_path, 'precision_id_' + name + '.png')
    success_path = os.path.join(plot_path, '_success.png_id_' + name + '.png')
    single_utils.plot_precision(gts, gts_test, precision_path)
    single_utils.plot_success(gts, gts_test, success_path)

if __name__ == '__main__':
    dnames = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13',
              'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
    data_dir = 'D:/DATASET/2DMOT2015/train/'
    dnames = sorted(os.listdir(data_dir), reverse=False)

    dnames = ['ADL-Rundle-8']

    gt_paths = []
    gt_test_paths = []
    plot_paths = []
    for dname in dnames:
        gt_paths.append('D:/DATASET/2DMOT2015/train/' + dname + '/gt/gt.txt')
        gt_test_paths.append('D:/DATASET/RESULT/2DMOT2015/' + dname + '/KCF_HOG/GT/result_gt_kcf_colorfeature.txt')  # result_gt_kcf_IOUfeature result_gt_kcf_deepfeature result_gt_kcf_colorfeature
        #gt_test_paths.append('D:/DATASET/RESULT/MOT16/DEEPSORT/result.txt')
        #gt_test_paths.append('D:/PROJECTS/sort-master/sort-master/output/' + dname + '.txt')
        plot_paths.append('D:/DATASET/RESULT/2DMOT2015/' + dname + '/KCF_HOG/PLOT')

    metrics_motchallenge_files(dnames, gt_paths, gt_test_paths)
    #metrics_single_for_first_detections(dnames, gt_paths, gt_test_paths, plot_paths)
