import cv2
import numpy as np
import os
from tracking_single.base import BaseCF
from tracking_multiple.mot_tracking.kcf_mot.base_tracking import Base_Tracking
from tracking_multiple.mot_helpers import utils

class ParticleTracker(Base_Tracking):
    def __init__(self):
        self.max_patch_size = 256
        super(ParticleTracker).__init__()

    def initiate(self, image, detection):
        x1, y1, w, h = detection.roi
        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4

        #self.hog = HOG((self.pw, self.ph))

        self.NUM_PARTICLES = 100
        np.random.seed(0)
        image_bound = image.shape
        # get_roi
        x, y, w, h = tuple(detection.roi)
        center = (x + w / 2, y + h / 2)
        w, h = int(round(w)), int(round(h))
        object_of_interest = cv2.getRectSubPix(image, (w, h), center)
        #object_of_interest = first_frame[y:y+h, x:x+w]

        self.object_bound = object_of_interest.shape
        self.cn = utils.calc_color_histogram(object_of_interest)
        #self.color_model = self.calc_hog(object_of_interest)

        self.s_t = np.random.rand(self.NUM_PARTICLES, 2)
        self.s_t[:, 0] *= image_bound[0]
        self.s_t[:, 1] *= image_bound[1]
        self.w_t = np.ones(self.NUM_PARTICLES) / self.NUM_PARTICLES  # Initially equal weights
        self.likelihood = np.zeros(self.NUM_PARTICLES)

    def predict(self, image):
        idxs = self.random_sample(self.w_t)
        idxs = self.random_sample(self.w_t)
        self.s_t = self.s_t[idxs, :]
        self.w_t = self.w_t[idxs]

        # Move particles according to motion model
        self.s_t = self.motion_model(self.s_t, image.shape, self.object_bound)

        # Compute appearance likelihood for each particle
        for j in range(self.NUM_PARTICLES):
            self.likelihood[j] = self.appearance_model(self.s_t[j, :], image, self.cn, self.object_bound)

        # Update particle weights
        self.w_t = self.w_t * self.likelihood
        self.w_t = self.w_t / np.sum(self.w_t)

        # Estimate object location based on weighted
        # states of the particles.
        estimate_t = (self.s_t.T.dot(self.w_t)).astype(int)
        res = np.array([estimate_t[1], estimate_t[0], self.object_bound[1], self.object_bound[0]])

        return res

    def update(self, image, detection):
        try:
            x, y, w, h = tuple(detection.roi)
            center = (x + w / 2, y + h / 2)
            w, h = int(round(w)), int(round(h))
            object_of_interest = cv2.getRectSubPix(image, (w, h), center)
            self.object_bound = object_of_interest.shape

            idxs = self.random_sample(self.w_t)
            self.s_t = self.s_t[idxs, :]
            self.w_t = self.w_t[idxs]
            # Move particles according to motion model
            self.s_t = self.motion_model(self.s_t, image.shape, self.object_bound)

            # Compute appearance likelihood for each particle
            for j in range(self.NUM_PARTICLES):
                self.likelihood[j] = self.appearance_model(self.s_t[j, :], image, self.cn, self.object_bound)

            # Update particle weights
            self.w_t = self.w_t * self.likelihood
            self.w_t = self.w_t / np.sum(self.w_t)

            return detection.roi
        except Exception as ex:
            print('Error in Particle : {}'.format(str(ex)))
            return 0, 0, 0, 0



    def calc_hog(self, patch):
        resized_image = cv2.resize(patch, (self.pw, self.ph), cv2.INTER_AREA)
        feature = self.hog.get_feature(resized_image)
        fc, fh, fw = feature.shape

        hann2t, hann1t = np.ogrid[0:fh, 0:fw]
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))
        hann2d = hann2t * hann1t
        feature = feature * hann2d

        return feature

    def extract_patch(self, s_t, image, object_bound):
        """ Extracts the part of the image corresponding to a given state. """
        s_t = s_t.astype(int)  # Convert state to discrete pixel locations
        return image[s_t[0]:s_t[0] + object_bound[0], s_t[1]:s_t[1] + object_bound[1]]

    def appearance_model(self, s_t, image, color_model, object_bound):
        # Hyperparameters
        l = 1  # lambda

        patch = self.extract_patch(s_t, image, object_bound)
        color_patch = utils.calc_color_histogram(patch)
        #color_patch = self.calc_hog(patch)

        divergence = utils.discrete_kl_divergence(color_model, color_patch)

        likelihood = np.exp(-l * divergence)
        return likelihood

    def motion_model(self, s_t, image_bound, object_bound):
        std_dev = 40

        # Motion estimation
        s_t = s_t + std_dev * np.random.randn(s_t.shape[0], 2)

        # Out-of-bounds check
        s_t[:, 0] = np.maximum(0, np.minimum(image_bound[0] - object_bound[0], s_t[:, 0]))
        s_t[:, 1] = np.maximum(0, np.minimum(image_bound[1] - object_bound[1], s_t[:, 1]))
        return s_t

    def random_sample(self, w_t):
        cumsum = np.cumsum(w_t)
        draws = np.random.rand(w_t.shape[0])
        idxs = np.zeros(w_t.shape[0])
        for i, draw in enumerate(draws):
            for j, probability in enumerate(cumsum):
                if probability > draw:
                    idxs[i] = j
                    break
        return idxs.astype(int)


class HOG():
    def __init__(self, winSize):
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nbins)

    def get_feature(self, image):
        winStride = self.winSize
        hist = self.hog.compute(image, winStride, padding=(0, 0))
        w, h = self.winSize
        sw, sh = self.blockStride
        w = w // sw - 1
        h = h // sh - 1
        return hist.reshape(w, h, 36).transpose(2, 1, 0)



