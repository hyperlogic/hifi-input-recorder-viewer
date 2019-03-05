# copied from hifi-ml/motion-matching/motion-matching.ipynb

import json
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import Hifi.recording
import Hifi.math

import importlib
importlib.reload(Hifi.recording)
importlib.reload(Hifi.math)

INPUT_RECORDING_FILENAME = 'matthew-stepping-no-turns3.json.gz'
POSE_NAMES = ['Hips', 'LeftFoot', 'RightFoot']
SUB_KEYS = {'angularVelocity': ['wx', 'wy', 'wz'],
            'rotation': ['rx', 'ry', 'rz', 'rw'],
            'translation': ['px', 'py', 'pz'],
            'velocity': ['dx', 'dy', 'dz']}
data = Hifi.recording.load(INPUT_RECORDING_FILENAME, POSE_NAMES, SUB_KEYS)

# transform Hips velocity into Hips local frame.
Hifi.recording.convert_to_relative_vector(data, 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                                          'Hips_dx', 'Hips_dy', 'Hips_dz')

# transform Hips angularVelocity into Hips local frame.
Hifi.recording.convert_to_relative_vector(data, 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                                          'LeftFoot_wx', 'LeftFoot_wy', 'LeftFoot_wz')

# transform LeftFoot position and rotation into Hips local frame.
Hifi.recording.convert_to_relative_xform(data, 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                                         'LeftFoot_px', 'LeftFoot_py', 'LeftFoot_pz', 'LeftFoot_rx', 'LeftFoot_ry', 'LeftFoot_rz', 'LeftFoot_rw')

# transform LeftFoot velocity into Hips local frame
Hifi.recording.convert_to_relative_vector(data, 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                                          'LeftFoot_dx', 'LeftFoot_dy', 'LeftFoot_dz')

# transform LeftFoot angularVelocity into Hips local frame
Hifi.recording.convert_to_relative_vector(data, 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                                          'LeftFoot_wx', 'LeftFoot_wy', 'LeftFoot_wz')

# transform RightFoot position and rotaiton to be relative to the Hips.
Hifi.recording.convert_to_relative_xform(data, 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                                         'RightFoot_px', 'RightFoot_py', 'RightFoot_pz', 'RightFoot_rx', 'RightFoot_ry', 'RightFoot_rz', 'RightFoot_rw')

# transform RightFoot velocity into Hips local frame
Hifi.recording.convert_to_relative_vector(data, 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                                          'RightFoot_dx', 'RightFoot_dy', 'RightFoot_dz')

# transform RightFoot angularVelocity into Hips local frame
Hifi.recording.convert_to_relative_vector(data, 'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                                          'RightFoot_wx', 'RightFoot_wy', 'RightFoot_wz')

ROOT_MOTION = True
if ROOT_MOTION:
    # transform Hips into world space
    Hifi.recording.apply_xform(data, 1.0, 'avatar_px', 'avatar_py', 'avatar_pz',
                               'avatar_rx', 'avatar_ry', 'avatar_rz', 'avatar_rw',
                               'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw')

    # transform Hips from world into sensor space
    Hifi.recording.apply_xform_inverse(data, 'sensor_s', 'sensor_px', 'sensor_py', 'sensor_pz',
                                       'sensor_rx', 'sensor_ry', 'sensor_rz', 'sensor_rw',
                                       'Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw')


# convert all rotations into expMaps
Hifi.recording.normalize_rotations(data, 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw')
Hifi.recording.convert_rotations_to_exp_map(data, 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw')
Hifi.recording.normalize_rotations(data, 'LeftFoot_rx', 'LeftFoot_ry', 'LeftFoot_rz', 'LeftFoot_rw')
Hifi.recording.convert_rotations_to_exp_map(data, 'LeftFoot_rx', 'LeftFoot_ry', 'LeftFoot_rz', 'LeftFoot_rw')
Hifi.recording.normalize_rotations(data, 'RightFoot_rx', 'RightFoot_ry', 'RightFoot_rz', 'RightFoot_rw')
Hifi.recording.convert_rotations_to_exp_map(data, 'RightFoot_rx', 'RightFoot_ry', 'RightFoot_rz', 'RightFoot_rw')

# prune data.
"""
data = data.drop(['Hips_px', 'Hips_py', 'Hips_pz', 'Hips_rx', 'Hips_ry', 'Hips_rz', 'Hips_rw',
                  'LeftFoot_rw', 'RightFoot_rw', 'Hips_valid', 'LeftFoot_valid', 'RightFoot_valid',
                  'avatar_px', 'avatar_py', 'avatar_pz', 'avatar_rx', 'avatar_ry', 'avatar_rz', 'avatar_rw',
                  'sensor_px', 'sensor_py', 'sensor_pz', 'sensor_rx', 'sensor_ry', 'sensor_rz', 'sensor_rw', 'sensor_s'], axis=1)
"""

print(data)
