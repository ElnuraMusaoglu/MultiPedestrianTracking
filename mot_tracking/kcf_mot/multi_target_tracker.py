# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from tracking_multiple.mot_tracking.kcf_mot.track import Track
from tracking_multiple.mot_tracking.kcf_mot import linear_assignment
from tracking_multiple.mot_tracking.kcf_mot import iou_matching
import numpy as np
from tracking_multiple.mot_tracking.kcf_mot.trackers.kcf import KCF_Tracker as Tracking
#from tracking_multiple.mot_tracking.kcf_mot.trackers.hcf import HCFTracker as Tracking
#from tracking_multiple.mot_tracking.kcf_mot.trackers.particle_filter_tracker import ParticleTracker as Tracking


class Tracker:
    def __init__(self, metric, success_criteria, max_iou_distance=0.7, max_age=10, n_init=4):  # max_age=30
        self.metric = metric
        self.success_criteria = success_criteria
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.tracks = []
        self._next_id = 1

    def initiate_for_first_detections(self, current_frame, detections):
        for detection in detections:
            self._initiate_track(1, current_frame, detection)

        return self._get_mot_metrics()

    def predict_for_first_detections(self, frame_id, current_frame):
        for track in self.tracks:
            track.predict_for_first_detections(frame_id, current_frame)

        return self._get_mot_metrics()

    def predict(self, frame_id, current_frame):
        for track in self.tracks:
            track.predict(frame_id, current_frame)

    def update_color(self, frame_id, current_frame, detections):  ## color_hists

        matches, unmatched_tracks, unmatched_detections = self._match_color(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(frame_id, current_frame, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(frame_id, current_frame, detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, color_hists, targets, targets_c = [], [], [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.deep_features
            color_hists += track.color_histograms
            targets += [track.track_id for _ in track.deep_features]
            targets_c += [track.track_id for _ in track.color_histograms]
            track.deep_features = []
            track.color_histograms = []

        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)
        self.metric.partial_fit2(np.asarray(color_hists), np.asarray(targets_c), active_targets)

        return self._get_mot_metrics()

    def update_deep(self, frame_id, current_frame, detections):  ## deep features

        matches, unmatched_tracks, unmatched_detections = self._match_deep(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(frame_id, current_frame, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(frame_id, current_frame, detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.deep_features
            targets += [track.track_id for _ in track.deep_features]
            track.deep_features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

        return self._get_mot_metrics()

    def update_iou(self, frame_id, current_frame, detections):  ##  IOU

        matches, unmatched_tracks, unmatched_detections = self._match_iou(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(frame_id, current_frame, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(frame_id, current_frame, detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        for track in self.tracks:
            if not track.is_confirmed():
                continue
            track.deep_features = []

        return self._get_mot_metrics()

    def _get_mot_metrics(self):
        mot_metrics = []
        for track in self.tracks:
            if not track.is_deleted() and track.is_confirmed():  ###   and track.is_confirmed()
                mot_metrics.append(track.get_mot_format())
        return mot_metrics

    def _match_color(self, detections):

        def gated_metric2(tracks, dets, track_indices, detection_indices):  # ELNURA
            color_hists = np.array([dets[i].color_hist for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance2(color_hists, targets)

            return cost_matrix

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using color and IOU features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric2, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _match_deep(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):  # ELNURA
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)

            return cost_matrix

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _match_iou(self, detections):

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        iou_track_candidates = confirmed_tracks + unconfirmed_tracks

        unmatched_detections = np.arange(len(detections))

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_b
        unmatched_tracks = list(set(unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, frame_id, current_frame, detection):
        tracking = Tracking()
        tracking.initiate(current_frame, detection.roi)
        self.tracks.append(Track(self._next_id, frame_id, self.n_init, self.max_age, detection.confidence,
                                 detection.feature, detection.color_hist, detection.roi, tracking))
        self._next_id += 1

