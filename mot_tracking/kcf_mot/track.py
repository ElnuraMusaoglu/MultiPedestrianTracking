
class TrackState:
    NewCreated = 1
    Confirmed = 2
    Occluded = 3
    Deleted = 4


class Track:
    def __init__(self, track_id, frame_id, n_init, max_age, detection_confidence,
                 deep_feature=None, color_hist=None, roi=None, tracking=None):
        self.track_id = track_id
        self.frame_id = frame_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.NewCreated
        self.deep_features = []
        self.color_histograms = []
        if color_hist is not None:
            self.color_histograms.append(color_hist)
        self.centers_with_frame_idx = {}
        self.detection_confidence = detection_confidence
        if deep_feature is not None:
            self.deep_features.append(deep_feature)
        if tracking is not None:
            self.tracking = tracking
        if roi is not None:
            self.roi = roi
        self._n_init = n_init
        self._max_age = max_age

    def predict(self, frame_id, current_frame):
        self.frame_id = frame_id
        self.roi = self.tracking.predict(current_frame)
        self.age += 1
        self.time_since_update += 1

    def predict_for_first_detections(self, frame_id, current_frame):
        self.frame_id = frame_id
        self.roi = self.tracking.predict(current_frame)
        self.age += 1
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.NewCreated and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def update(self, frame_id, current_frame, detection):
        self.frame_id = frame_id
        self.deep_features.append(detection.feature)
        self.color_histograms.append(detection.color_hist)
        self.roi = self.tracking.update(current_frame, detection)
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.NewCreated and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def get_mot_format(self):
        mot_tuple = (
            self.frame_id, self.track_id, self.roi[0], self.roi[1], self.roi[2], self.roi[3], self.detection_confidence,
            -1, -1, -1
        )
        return mot_tuple

    def mark_missed(self):
        if self.state == TrackState.NewCreated:
            self.state = TrackState.Deleted   ## Elnura
            #self.state = TrackState.Confirmed
            #return
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
        #else:
            #self.state = TrackState.Occluded

    def is_new_created(self):
        return self.state == TrackState.NewCreated

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_occluded(self):
        return self.state == TrackState.Occluded

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def set_roi(self, roi):
        self.roi = roi

    def get_roi(self):
        return self.roi
