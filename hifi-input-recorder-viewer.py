#!/usr/bin/env python

import sys
import math

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider, QWidget)

import OpenGL.GL as gl

import Hifi.recording
import Hifi.math

import functools
import operator
import pandas



##############################################
## paste jupyter notebook here

# INPUT_RECORDING_FILENAME = "53_Matthew_Rockettes-Kick.gz"
INPUT_RECORDING_FILENAME = "motion-matching/matthew-stepping-no-turns3.json.gz"
POSE_NAMES = ['Head', 'Hips', 'LeftFoot', 'RightFoot', 'LeftHand', 'RightHand']
SUB_KEYS = {'angularVelocity': ['wx', 'wy', 'wz'],
            'rotation': ['rx', 'ry', 'rz', 'rw'],
            'translation': ['px', 'py', 'pz'],
            'velocity': ['dx', 'dy', 'dz']}
frame = 0
data = Hifi.recording.load(INPUT_RECORDING_FILENAME, POSE_NAMES, SUB_KEYS)

import scipy.signal

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)

    # compute group delay of filter
    w, gd = scipy.signal.group_delay((b, a))

    # shift data by maximum group delay, canceling out the delay
    fudge_factor = 0.75
    shift_amount = int(gd.max() * fudge_factor)  # in samples
    data = data.shift(-shift_amount)

    # filter the shifted data
    y = scipy.signal.lfilter(b, a, data)
    return y

def lowpass_filter(array):
    # Filter requirements.
    order = 3
    fs = 90.0       # sample rate, Hz
    cutoff = 0.5  # desired cutoff frequency of the filter, Hz
    return butter_lowpass_filter(array, cutoff, fs, order)

if 'sensor_px' in data:
    print("new recording")
    # transform data from avatar to sensor space.
    for pose in POSE_NAMES:
        print("transforming " + pose + " into sensor space")
        Hifi.recording.apply_xform(data, 1.0, 'avatar_px', 'avatar_py', 'avatar_pz', 'avatar_rx', 'avatar_ry', 'avatar_rz', 'avatar_rw',
                                   pose + '_px', pose + '_py', pose + '_pz', pose + '_rx', pose + '_ry', pose + '_rz', pose + '_rw')
        Hifi.recording.apply_xform_inverse(data, 'sensor_s', 'sensor_px', 'sensor_py', 'sensor_pz', 'sensor_rx', 'sensor_ry', 'sensor_rz', 'sensor_rw',
                                           pose + '_px', pose + '_py', pose + '_pz', pose + '_rx', pose + '_ry', pose + '_rz', pose + '_rw')
else:
    print("old recording")

# filter test.
data["Hips2_px"] = lowpass_filter(data["Hips_px"])
data["Hips2_py"] = lowpass_filter(data["Hips_py"])
data["Hips2_pz"] = lowpass_filter(data["Hips_pz"])
data["Hips2_rx"] = data["Hips_rx"]
data["Hips2_ry"] = data["Hips_ry"]
data["Hips2_rz"] = data["Hips_rz"]
data["Hips2_rw"] = data["Hips_rw"]
POSE_NAMES.append("Hips2")

##############################################

def gluPerspective(fovy, aspect, nearVal, farVal):
    f = 1.0 / math.tan(fovy / 2.0)
    a = (farVal + nearVal) / (nearVal - farVal)
    b = (2 * farVal * nearVal) / (nearVal - farVal)
    m = [f / aspect, 0, 0, 0,
         0,          f, 0, 0,
         0,          0, a, -1,
         0,          0, b, 0]
    gl.glLoadMatrixd(m)

def drawPose(name, frame):
    global data

    DRAW_TRAILS = True
    TRAIL_LENGTH = 30
    TRAIL_COLOR = [0.0, 0.5, 0.5, 1.0]

    pos = Hifi.math.Vec3(data[name + "_px"][frame], data[name + "_py"][frame], data[name + "_pz"][frame])
    rot = Hifi.math.Quat(data[name + "_rx"][frame], data[name + "_ry"][frame], data[name + "_rz"][frame], data[name + "_rw"][frame])
    xAxis = pos + rot.rotate(Hifi.math.Vec3(0.1, 0.0, 0.0))
    yAxis = pos + rot.rotate(Hifi.math.Vec3(0.0, 0.1, 0.0))
    zAxis = pos + rot.rotate(Hifi.math.Vec3(0.0, 0.0, 0.1))
    gl.glBegin(gl.GL_LINES)
    gl.glColor4d(1.0, 0.0, 0.0, 1.0)
    gl.glVertex3d(pos.x, pos.y, pos.z)
    gl.glVertex3d(xAxis.x, xAxis.y, xAxis.z)
    gl.glColor4f(0.0, 1.0, 0.0, 1.0)
    gl.glVertex3d(pos.x, pos.y, pos.z)
    gl.glVertex3d(yAxis.x, yAxis.y, yAxis.z)
    gl.glColor4f(0.0, 0.0, 1.0, 1.0)
    gl.glVertex3d(pos.x, pos.y, pos.z)
    gl.glVertex3d(zAxis.x, zAxis.y, zAxis.z)
    gl.glEnd()
    if DRAW_TRAILS:
        gl.glBegin(gl.GL_LINE_STRIP)
        for i in range(TRAIL_LENGTH):
            frameIndex = max(frame - i, 0)
            shade = ((TRAIL_LENGTH - i) / TRAIL_LENGTH)
            pos = Hifi.math.Vec3(data[name + "_px"][frameIndex], data[name + "_py"][frameIndex], data[name + "_pz"][frameIndex])
            gl.glColor4d(TRAIL_COLOR[0] * shade, TRAIL_COLOR[1] * shade, TRAIL_COLOR[2] * shade, TRAIL_COLOR[3])
            gl.glVertex3d(pos.x, pos.y, pos.z)
        gl.glEnd()

def drawFloor(height):
    gridSpacing = 0.2
    numLines = 10
    end = (numLines * gridSpacing) / 2.0
    start = -end
    gl.glColor4d(0.0, 0.5, 0.0, 0.0)
    gl.glBegin(gl.GL_LINES)

    # draw grid lines
    for i in range(numLines + 1):
        x = start + (i * gridSpacing)
        gl.glVertex3d(x, 0.0, start);
        gl.glVertex3d(x, 0.0, end);
        gl.glVertex3d(start, 0.0, x);
        gl.glVertex3d(end, 0.0, x);
    gl.glEnd()

class Window(QWidget):

    def __init__(self):
        super(Window, self).__init__()
        self.glWidget = GLWidget()
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        self.setLayout(mainLayout)
        self.setWindowTitle("Hifi Input Recorder Viewer")

class GLWidget(QOpenGLWidget):
    xRotationChanged = pyqtSignal(int)
    yRotationChanged = pyqtSignal(int)
    zRotationChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.boomLength = 5
        self.xOffset = 0
        self.yOffset = 0

        self.lastPos = QPoint()

        self.trolltechGreen = QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        self.trolltechPurple = QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

    def getOpenglInfo(self):
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
        """.format(
            gl.glGetString(gl.GL_VENDOR),
            gl.glGetString(gl.GL_RENDERER),
            gl.glGetString(gl.GL_VERSION),
            gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        )

        return info

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            self.xRotationChanged.emit(angle)
            self.update()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            self.yRotationChanged.emit(angle)
            self.update()

    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            self.zRotationChanged.emit(angle)
            self.update()

    def initializeGL(self):
        print(self.getOpenglInfo())

        # black
        self.setClearColor(QColor.fromRgbF(0.0, 0.0, 0.0))
        gl.glShadeModel(gl.GL_FLAT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glTranslated(self.xOffset, self.yOffset, -self.boomLength)
        gl.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        gl.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        gl.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)

        global frame
        drawFloor(0.0)
        for name in POSE_NAMES:
            drawPose(name, frame)
        frame = (frame + 1) % len(data)
        self.update()

    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return

        gl.glViewport((width - side) // 2, (height - side) // 2, side,
                           side)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        #gl.glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        gluPerspective(30.0 * math.pi / 180, 1, 0.01, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setYRotation(self.yRot + 8 * dx)
        elif event.buttons() & Qt.MiddleButton:
            self.xOffset = self.xOffset + (0.01 * dx)
            self.yOffset = self.yOffset - (0.01 * dy)
        self.lastPos = event.pos()

    def wheelEvent(self, event):
        self.boomLength = self.boomLength - (event.angleDelta().y() / 120.0)

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def setClearColor(self, c):
        gl.glClearColor(c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def setColor(self, c):
        gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
