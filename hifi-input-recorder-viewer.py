#!/usr/bin/env python

import sys
import math
import numpy
import pandas

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider, QWidget)

import OpenGL.GL as gl

import Hifi.recording
import Hifi.vecmath

##############################################
## paste jupyter notebook here

GENERATE_OUTPUT = True

trajectory_frames = [18, 36, 62, 90, -18, -36, -62, -90]
num_trajectory_points = len(trajectory_frames)

# which poses to render
POSE_NAMES = ['Head', 'Hips', 'LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Root']
POSE_NAMES = POSE_NAMES + ["RootTraj{}".format(i + 1) for i in range(num_trajectory_points)]
# POSE_NAMES = POSE_NAMES + ["HeadTraj{}".format(i + 1) for i in range(num_trajectory_points)]

if GENERATE_OUTPUT:

    # load all stepping data and concatenate it together
    microsteps = Hifi.recording.prepare_data_for_motion_matching("motion-matching/microsteps-no-turns.json.gz", trajectory_frames)
    stepping1 = Hifi.recording.prepare_data_for_motion_matching("motion-matching/matthew-stepping-no-turns1.json.gz", trajectory_frames)
    stepping2 = Hifi.recording.prepare_data_for_motion_matching("motion-matching/matthew-stepping-no-turns2.json.gz", trajectory_frames)
    stepping3 = Hifi.recording.prepare_data_for_motion_matching("motion-matching/matthew-stepping-no-turns3.json.gz", trajectory_frames)
    data = microsteps.append(stepping1.append(stepping2, ignore_index = True).append(stepping3, ignore_index = True), ignore_index = True)

    # load stepping and micro steps
    # microsteps = Hifi.recording.prepare_data_for_motion_matching("motion-matching/microsteps-no-turns.json.gz", trajectory_frames)
    # stepping3 = Hifi.recording.prepare_data_for_motion_matching("motion-matching/matthew-stepping-no-turns3.json.gz", trajectory_frames)
    # data = stepping3.append(microsteps, ignore_index = True)

    # just load one recording.
    # data = Hifi.recording.prepare_data_for_motion_matching("motion-matching/matthew-stepping-no-turns3.json.gz", trajectory_frames)

    # reset all the frame numbers
    data['id'] = pandas.Series(numpy.arange(len(data)))

    # keep only the keys we care about.
    output_keys = ['id', 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                   'LeftFoot_px', 'LeftFoot_py', 'LeftFoot_pz', 'LeftFoot_rx', 'LeftFoot_ry', 'LeftFoot_rz', 'LeftFoot_rw',
                   'RightFoot_px', 'RightFoot_py', 'RightFoot_pz', 'RightFoot_rx', 'RightFoot_ry', 'RightFoot_rz', 'RightFoot_rw',
                   'Head_px', 'Head_py', 'Head_pz', 'Head_rx', 'Head_ry', 'Head_rz', 'Head_rw',
                   'LeftHand_px', 'LeftHand_py', 'LeftHand_pz', 'LeftHand_rx', 'LeftHand_ry', 'LeftHand_rz', 'LeftHand_rw',
                   'RightHand_px', 'RightHand_py', 'RightHand_pz', 'RightHand_rx', 'RightHand_ry', 'RightHand_rz', 'RightHand_rw']
    for i in range(num_trajectory_points):
        output_keys.append("RootTraj{}_px".format(i + 1))
        output_keys.append("RootTraj{}_py".format(i + 1))
        output_keys.append("RootTraj{}_pz".format(i + 1))
    for i in range(num_trajectory_points):
        output_keys.append("HeadTraj{}_px".format(i + 1))
        output_keys.append("HeadTraj{}_py".format(i + 1))
        output_keys.append("HeadTraj{}_pz".format(i + 1))

    Hifi.recording.output_data_for_motion_matching(output_keys, data)

else:

    # "53_Matthew_Rockettes-Kick.gz"
    data = Hifi.recording.prepare_data_for_visualization("motion-matching/matthew-stepping-no-turns3.json.gz", trajectory_frames)


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

    pos = Hifi.vecmath.Vec3(data[name + "_px"][frame], data[name + "_py"][frame], data[name + "_pz"][frame])
    rot = Hifi.vecmath.Quat(data[name + "_rx"][frame], data[name + "_ry"][frame], data[name + "_rz"][frame], data[name + "_rw"][frame])
    xAxis = pos + rot.rotate(Hifi.vecmath.Vec3(0.1, 0.0, 0.0))
    yAxis = pos + rot.rotate(Hifi.vecmath.Vec3(0.0, 0.1, 0.0))
    zAxis = pos + rot.rotate(Hifi.vecmath.Vec3(0.0, 0.0, 0.1))
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
            pos = Hifi.vecmath.Vec3(data[name + "_px"][frameIndex], data[name + "_py"][frameIndex], data[name + "_pz"][frameIndex])
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
