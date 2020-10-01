#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import numpy as np
import sys


class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self.browse_button.clicked.connect(self.SLOT_browse_button)

        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        # Establishing algorithms
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()

        self.template_path = None
        self.template_img = None
        self.template_kp = None
        self.template_desc = None
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.SIFT = cv2.xfeatures2d.SIFT_create()

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        self.template_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        self.template_kp, self.template_desc = self.SIFT.detectAndCompute(self.template_img, None)

        pixmap = self.convert_cv_to_pixmap(self.template_img)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

    def convert_cv_to_pixmap(self, cv_img, is_homography=False):
        if is_homography:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp_grayframe, desc_grayframe = self.SIFT.detectAndCompute(grayframe, None)
        matches = self.flann.knnMatch(self.template_desc, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6*n.distance:
                good_points.append(m)

        # Homography

        if len(good_points) > 10:
            template_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            cam_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(template_pts, cam_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # Perspective Transform
            h, w = self.template_img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            img3 = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        else:
            img3 = cv2.drawMatches(self.template_img, self.template_kp, grayframe, kp_grayframe, good_points, grayframe)

        pixmap = self.convert_cv_to_pixmap(img3, is_homography=True)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
