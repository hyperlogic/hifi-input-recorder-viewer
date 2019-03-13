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
import numpy

##############################################
## paste jupyter notebook here

import scipy.signal
import json

GENERATE_OUTPUT = True
NAN_CHECK = True

def nan_check(data):
    if NAN_CHECK:
        for key in data.keys():
            if data[key].isnull().values.any():
                print("WARNING: data['{}'] has {} NaNs it!".format(key, data[key].isnull().sum()))

def dump_data(keys, data):
    newData = pandas.DataFrame()
    for key in keys:
        newData[key] = data[key]

    with open('data.json', 'w') as outfile:
        json.dump(newData.values.tolist(), outfile, indent = 4)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):

    # create filter
    b, a = butter_lowpass(cutoff, fs, order=order)

    # compute group delay of filter
    w, gd = scipy.signal.group_delay((b, a))

    # pad both ends of data
    # this is necessary for the filtering to be smooth
    padding_count = 180
    intro_padding = pandas.Series(data[0], index=numpy.arange(padding_count))
    outro_padding = pandas.Series(data[len(data) - 1], index=numpy.arange(padding_count))
    padded_data = intro_padding.append(data, ignore_index = True).append(outro_padding, ignore_index = True)

    y = scipy.signal.lfilter(b, a, padded_data)
    result = pandas.Series(y)

    # compute latency introduced by the filter
    fudge_factor = 0.75
    shift_amount = int(gd.max() * fudge_factor)  # in samples

    # clip out padding and cancel out filter latency by shifting data
    clipped_result = result[(padding_count + shift_amount):-(padding_count - shift_amount)]
    clipped_result.index = numpy.arange(len(clipped_result))
    return clipped_result

# INPUT_RECORDING_FILENAME = "53_Matthew_Rockettes-Kick.gz"

# perhaps needs lower filter cuttoff, root motion is very sinusoidal
# INPUT_RECORDING_FILENAME = "motion-matching/microsteps-no-turns.json.gz"

# INPUT_RECORDING_FILENAME = "motion-matching/matthew-stepping-no-turns1.json.gz"

# back and forth
# INPUT_RECORDING_FILENAME = "motion-matching/matthew-stepping-no-turns2.json.gz"

# circles
INPUT_RECORDING_FILENAME = "motion-matching/matthew-stepping-no-turns3.json.gz"

POSE_NAMES = ['Head', 'Hips', 'LeftFoot', 'RightFoot', 'LeftHand', 'RightHand']
SUB_KEYS = {'angularVelocity': ['wx', 'wy', 'wz'],
            'rotation': ['rx', 'ry', 'rz', 'rw'],
            'translation': ['px', 'py', 'pz'],
            'velocity': ['dx', 'dy', 'dz']}

print("Loading recording", INPUT_RECORDING_FILENAME)
data = Hifi.recording.load(INPUT_RECORDING_FILENAME, POSE_NAMES, SUB_KEYS)

nan_check(data)

#
# Transform all poses into sensor space, if possible
#

if 'sensor_px' in data:
    # transform data from avatar to sensor space.
    for pose in POSE_NAMES:
        print("transforming " + pose + " into sensor space")
        Hifi.recording.apply_xform(data, 1.0, 'avatar_px', 'avatar_py', 'avatar_pz', 'avatar_rx', 'avatar_ry', 'avatar_rz', 'avatar_rw',
                                   pose + '_px', pose + '_py', pose + '_pz', pose + '_rx', pose + '_ry', pose + '_rz', pose + '_rw')
        Hifi.recording.apply_xform_inverse(data, 'sensor_s', 'sensor_px', 'sensor_py', 'sensor_pz', 'sensor_rx', 'sensor_ry', 'sensor_rz', 'sensor_rw',
                                           pose + '_px', pose + '_py', pose + '_pz', pose + '_rx', pose + '_ry', pose + '_rz', pose + '_rw')
        nan_check(data)
else:
    print("Recording is in avatar space")

#
# Filter settings
#

filter_order = 3
filter_fs = 90.0       # sample rate, Hz
filter_cutoff = 0.5  # desired cutoff frequency of the filter, Hz

#
# compute motion of root by filtering the motion of the hips.
#

print("Filtering hips to generate root motion")

root_y = data["Hips_py"].mean()
num_samples = len(data["Hips_py"])
data["Root_px"] = butter_lowpass_filter(data["Hips_px"], filter_cutoff, filter_fs, filter_order)
data["Root_py"] = pandas.Series([root_y for i in range(num_samples)])
data["Root_pz"] = butter_lowpass_filter(data["Hips_pz"], filter_cutoff, filter_fs, filter_order)

#
# compute rotation of root by filtering rotation of the hips
#

thetas = []
z_axis = Hifi.math.Vec3(0, 0, 1)
y_axis = Hifi.math.Vec3(0, 1, 0)
for i in range(num_samples):
    rot = Hifi.math.Quat(data["Hips_rx"][i], data["Hips_ry"][i], data["Hips_rz"][i], data["Hips_rw"][i])
    forward = rot.rotate(z_axis)
    new_theta = math.atan2(forward.x, forward.z)

    if i > 0:
        prev_theta = thetas[i - 1]
    else:
        prev_theta = math.atan2(forward.x, forward.z)

    # shift theta so that it is less then 180 degrees from the previous theta
    # this is to prevent discontinuities and keep rotations smooth.
    while (new_theta - prev_theta) > math.pi:
        new_theta = new_theta - (2.0 * math.pi)
    while (new_theta - prev_theta) < -math.pi:
        new_theta = new_theta + (2.0 * math.pi)
    thetas.append(new_theta)

filtered_thetas = butter_lowpass_filter(pandas.Series(thetas), filter_cutoff, filter_fs, filter_order)
rot_quats = [Hifi.math.Quat.fromAngleAxis(theta, y_axis) for theta in filtered_thetas]

data["Root_rx"] = pandas.Series([q.x for q in rot_quats])
data["Root_ry"] = pandas.Series([q.y for q in rot_quats])
data["Root_rz"] = pandas.Series([q.z for q in rot_quats])
data["Root_rw"] = pandas.Series([q.w for q in rot_quats])
POSE_NAMES.append("Root")

nan_check(data)

#
# compute trajectories of root motion
#

num_trajectory_points = 4
trajectory_frames = [18, 36, 62, 90]
def timeShift(series, offset):
    num_samples = len(series)
    return [series[max(0, min(num_samples - 1, i + offset))] for i in range(num_samples)]

# generate forward trajectories
suffixes = ["_px", "_py", "_pz", "_rx", "_ry", "_rz", "_rw"]

for i in range(num_trajectory_points):
    for suffix in suffixes:
        data["RootF{}{}".format(i + 1, suffix)] = timeShift(data["Root{}".format(suffix)], trajectory_frames[i])
    POSE_NAMES.append("RootF{}".format(i + 1))

#
# Transform everything back into Root relative space
#

if GENERATE_OUTPUT:
    output_poses = ['Hips', 'LeftFoot', 'RightFoot', 'Head', 'LeftHand', 'RightHand']
    for i in range(num_trajectory_points):
        output_poses.append("RootF{}".format(i + 1))

    for pose in output_poses:
        print("transforming " + pose + " into Root relative space")

        Hifi.recording.apply_xform_inverse(data, 1.0, 'Root_px', 'Root_py', 'Root_pz', 'Root_rx', 'Root_ry', 'Root_rz', 'Root_rw',
                                           pose + '_px', pose + '_py', pose + '_pz', pose + '_rx', pose + '_ry', pose + '_rz', pose + '_rw')

    # keep only the keys we care about.
    output_keys = ['id', 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                   'LeftFoot_px', 'LeftFoot_py', 'LeftFoot_pz', 'LeftFoot_rx', 'LeftFoot_ry', 'LeftFoot_rz', 'LeftFoot_rw',
                   'RightFoot_px', 'RightFoot_py', 'RightFoot_pz', 'RightFoot_rx', 'RightFoot_ry', 'RightFoot_rz', 'RightFoot_rw',
                   'Head_px', 'Head_py', 'Head_pz', 'Head_rx', 'Head_ry', 'Head_rz', 'Head_rw',
                   'LeftHand_px', 'LeftHand_py', 'LeftHand_pz', 'LeftHand_rx', 'LeftHand_ry', 'LeftHand_rz', 'LeftHand_rw',
                   'RightHand_px', 'RightHand_py', 'RightHand_pz', 'RightHand_rx', 'RightHand_ry', 'RightHand_rz', 'RightHand_rw']
    for i in range(num_trajectory_points):
        output_keys.append("RootF{}_px".format(i + 1))
        output_keys.append("RootF{}_py".format(i + 1))
        output_keys.append("RootF{}_pz".format(i + 1))

    dump_data(output_keys, data)

    # print c++ enum
    print("enum RowIndices {")
    for i in range(len(output_keys)):
        key = output_keys[i]
        if i == 0:
            print("    {}_INDEX = 0,".format(key.upper()))
        elif i == len(output_keys) - 1:
            print("    {}_INDEX".format(key.upper()))
        else:
            print("    {}_INDEX,".format(key.upper()))
    print("};")


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
    TRAIL_LENGTH = 45
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

    def keyPressEvent(self, event):
        self.glWidget.keyPressEvent(event)


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
        self.frame = 0
        self.paused = False

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
        return QSize(800, 800)

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
        # black
        self.setClearColor(QColor.fromRgbF(0.0, 0.0, 0.0))
        gl.glShadeModel(gl.GL_FLAT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    def incrementFrame(self):
        self.frame = (self.frame + 1) % len(data)

    def decrementFrame(self):
        self.frame = (self.frame - 1) % len(data)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glTranslated(self.xOffset, self.yOffset, -self.boomLength)
        gl.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        gl.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        gl.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)

        drawFloor(0.0)
        for name in POSE_NAMES:
            drawPose(name, self.frame)

        if not self.paused:
            self.incrementFrame()

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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            # restart
            self.frame = 0
        elif event.key() == Qt.Key_Space:
            # toggle pause
            self.paused = not self.paused
        elif self.paused and event.key() == Qt.Key_Period:
            self.incrementFrame()
        elif self.paused and event.key() == Qt.Key_Comma:
            self.decrementFrame()

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
