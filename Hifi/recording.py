# Hifi input recording loader
import json
import gzip
import pandas
from . import math
from distutils.version import LooseVersion

# INPUT_RECORDING_FILENAME = 'hifi-input-recordings/matthew-stomping-flick-hips.json.gz'
# POSE_NAMES = ['Head', 'LeftHand', 'RightHand', 'Hips', 'LeftFoot', 'RightFoot']
# SUB_KEYS = {'angularVelocity': ['wx', 'wy', 'wz'],
#             'rotation': ['rx', 'ry', 'rz', 'rw'],
#             'translation': ['px', 'py', 'pz'],
#             'velocity': ['dx', 'dy', 'dz']}
# data = load_input_recording(INPUT_RECORDING_FILENAME, POSE_NAMES, SUB_KEYS)
def load(filename, pose_names, sub_keys):
    """
Returns a pandas.DataFrame object, that contain the columns indicated by pose_names, and each row corresponds to a single frame in the recording.
The DataFrame object will also contain the avatar and sensor matrices if available in the recording.
avatar_px, avatar_py, avatar_pz, avatar_rx, avatar_ry, avatar_rz, avatar_rw
sensor_px, sensor_py, sensor_pz, sensor_rx, sensor_ry, sensor_rz, sensor_rw, sensor_s
    """
    with gzip.open(filename, 'rb') as f:
        json_data = json.loads(f.read())
        frame_count = json_data['frameCount']
        pose_list = json_data['poseList']

        # use the first frame of data to map a pose_name to an index
        pose_name_to_index = {}
        for name in pose_names:
            match = [i for i in range(len(pose_list[0])) if name == pose_list[0][i]['name']]
            if len(match) == 1:
                pose_name_to_index[name] = match[0]

        # build a row of all the column names
        # TODO: not sure if id is strictly necessary
        key_row = ['id']
        for name in pose_names:
            for key in pose_list[0][pose_name_to_index[name]]['pose'].keys():
                if key in sub_keys.keys():
                    [key_row.append(name + '_' + subkey) for subkey in sub_keys[key]]
                else:
                    key_row.append(name + '_' + key)

        data = []
        frame_count = 0

        # now insert all the column values
        for frame in pose_list:
            row = [frame_count]
            frame_count = frame_count + 1
            for name in pose_names:
                for key in frame[pose_name_to_index[name]]['pose'].keys():
                    if key in sub_keys.keys():
                        [row.append(frame[pose_name_to_index[name]]['pose'][key][i]) for i in range(len(sub_keys[key]))]
                    else:
                        row.append(frame[pose_name_to_index[name]]['pose'][key])
            data.append(row)

        INPUT_CALIBRATION_VERSION = LooseVersion("0.1")
        if ('version' in json_data) and (LooseVersion(json_data['version']) >= INPUT_CALIBRATION_VERSION):
            i = 0
            for frame in json_data['inputCalibrationDataList']:
                avatar_pos = frame['avatar']['translation']
                avatar_rot = frame['avatar']['rotation']
                sensor_pos = frame['sensorToWorld']['translation']
                sensor_rot = frame['sensorToWorld']['rotation']
                sensor_scale = frame['sensorToWorld']['scale']
                data[i].extend(avatar_pos + avatar_rot + sensor_pos + sensor_rot + [sensor_scale[0]])
                i = i + 1
            key_row.extend(['avatar_px', 'avatar_py', 'avatar_pz', 'avatar_rx', 'avatar_ry', 'avatar_rz', 'avatar_rw',
                            'sensor_px', 'sensor_py', 'sensor_pz', 'sensor_rx', 'sensor_ry', 'sensor_rz', 'sensor_rw', 'sensor_s'])

        return pandas.DataFrame.from_records(data, columns=key_row)

def normalize_rotations(data, x_key, y_key, z_key, w_key):
    """Mutates the quaternion rotations passed in from a pandas.DataFrame to have the same polarity as the previous frame"""
    prev_q = math.Quat(0.0, 0.0, 0.0, 1.0)
    for i in range(len(data)):
        q = math.Quat(data.at[i, x_key], data.at[i, y_key], data.at[i, z_key], data.at[i, w_key])
        if q.dot(prev_q) < 0:
            prev_q = -q
            # flip the sign of the quat
            data.at[i, x_key] = prev_q.x
            data.at[i, y_key] = prev_q.y
            data.at[i, z_key] = prev_q.z
            data.at[i, w_key] = prev_q.w
        else:
            prev_q = q

def convert_rotations_to_exp_map(data, x_key, y_key, z_key, w_key):
    """Mutates the quaternion rotations passed in from a pandas.DataFrame to be exponental maps aka quat-logs"""
    prev_rot = math.Quat(0.0, 0.0, 0.0, 1.0)
    for i in range(len(data)):
        q = math.Quat(data.at[i, x_key], data.at[i, y_key], data.at[i, z_key], data.at[i, w_key])
        if q.dot(prev_rot) < 0:
            q = -q
        prev_rot = q
        q = q.log()
        data.at[i, x_key] = q.x
        data.at[i, y_key] = q.y
        data.at[i, z_key] = q.z
        data.at[i, w_key] = q.w

def apply_xform(data, xform_scale, xform_px, xform_py, xform_pz,
                xform_rx, xform_ry, xform_rz, xform_rw,
                px, py, pz, rx, ry, rz, rw):

    """Mutates the position and rotaiton columns passed in from a pandas.DataFrame by the given xform"""
    for i in range(len(data)):

        if isinstance(xform_scale, str):
            xform_s = data.at[i, xform_scale]
        else:
            xform_s = xform_scale

        xform_pos = math.Vec3(data.at[i, xform_px], data.at[i, xform_py], data.at[i, xform_pz])
        xform_rot = math.Quat(data.at[i, xform_rx], data.at[i, xform_ry], data.at[i, xform_rz], data.at[i, xform_rw])
        xform = math.Xform(xform_pos, xform_rot)

        pos = math.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])
        rot = math.Quat(data.at[i, rx], data.at[i, ry], data.at[i, rz], data.at[i, rw])

        newPos = xform.xformPoint(pos * xform_s)
        newRot = xform.rot * rot

        data.at[i, px] = newPos.x
        data.at[i, py] = newPos.y
        data.at[i, pz] = newPos.z

        data.at[i, rx] = newRot.x
        data.at[i, ry] = newRot.y
        data.at[i, rz] = newRot.z
        data.at[i, rw] = newRot.w

def apply_xform_inverse(data, xform_scale, xform_px, xform_py, xform_pz,
                        xform_rx, xform_ry, xform_rz, xform_rw,
                        px, py, pz, rx, ry, rz, rw):
    """Mutates the position and rotaiton columns passed in from a pandas.DataFrame by the inverse of the given xform"""
    for i in range(len(data)):

        if isinstance(xform_scale, str):
            xform_s = 1.0 / data.at[i, xform_scale]
        else:
            xform_s = 1.0 / xform_scale

        xform_pos = math.Vec3(data.at[i, xform_px], data.at[i, xform_py], data.at[i, xform_pz])
        xform_rot = math.Quat(data.at[i, xform_rx], data.at[i, xform_ry], data.at[i, xform_rz], data.at[i, xform_rw])
        xform = math.Xform(xform_pos, xform_rot)
        xform = xform.inverse()

        pos = math.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])
        rot = math.Quat(data.at[i, rx], data.at[i, ry], data.at[i, rz], data.at[i, rw])

        newPos = xform.xformPoint(pos) * xform_s
        newRot = xform.rot * rot

        data.at[i, px] = newPos.x
        data.at[i, py] = newPos.y
        data.at[i, pz] = newPos.z

        data.at[i, rx] = newRot.x
        data.at[i, ry] = newRot.y
        data.at[i, rz] = newRot.z
        data.at[i, rw] = newRot.w

def apply_xform_vector(data, xform_rx, xform_ry, xform_rz, xform_rw, px, py, pz):
    """Mutates the position columns passed in from a pandas.DataFrame by the given xform"""
    for i in range(len(data)):
        xform_rot = math.Quat(data.at[i, xform_rx], data.at[i, xform_ry], data.at[i, xform_rz], data.at[i, xform_rw])

        pos = math.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])

        newPos = xform_rot.rotate(pos)

        data.at[i, px] = newPos.x
        data.at[i, py] = newPos.y
        data.at[i, pz] = newPos.z

def apply_xform_inverse_vector(data, xform_rx, xform_ry, xform_rz, xform_rw, px, py, pz):
    """Mutates the position columns passed in from a pandas.DataFrame by the inverse of the given xform"""
    for i in range(len(data)):
        xform_rot = math.Quat(data.at[i, xform_rx], data.at[i, xform_ry], data.at[i, xform_rz], data.at[i, xform_rw])
        xform_rot = xform_rot.inverse()

        pos = math.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])

        newPos = xform_rot.rotate(pos)

        data.at[i, px] = newPos.x
        data.at[i, py] = newPos.y
        data.at[i, pz] = newPos.z

def convert_to_relative_xform(data, head_px, head_py, head_pz,
                              head_rx, head_ry, head_rz, head_rw,
                              px, py, pz, rx, ry, rz, rw):
    """Mutates the position and rotaiton columns passed in from a pandas.DataFrame to relative to the horizontal head"""

    for i in range(len(data)):
        headPos = math.Vec3(data.at[i, head_px], data.at[i, head_py], data.at[i, head_pz])
        headRot = math.Quat(data.at[i, head_rx], data.at[i, head_ry], data.at[i, head_rz], data.at[i, head_rw])

        pos = math.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])
        rot = math.Quat(data.at[i, rx], data.at[i, ry], data.at[i, rz], data.at[i, rw])

        # compute head rotation with pitch and roll removed
        swingRot, twistRot = headRot.swingTwistDecomposition(math.Vec3(0, 1, 0))
        invHead = math.Xform(headPos, twistRot)
        invHead = invHead.inverse()
        newPos = invHead.xformPoint(pos)
        newRot = invHead.rot * rot

        data.at[i, px] = newPos.x
        data.at[i, py] = newPos.y
        data.at[i, pz] = newPos.z

        data.at[i, rx] = newRot.x
        data.at[i, ry] = newRot.y
        data.at[i, rz] = newRot.z
        data.at[i, rw] = newRot.w


def convert_to_relative_vector(data, head_px, head_py, head_pz,
                               head_rx, head_ry, head_rz, head_rw,
                               vx, vy, vz):
    """Mutates the vector columns passed in from a pandas.DataFrame to relative to the horizontal head"""

    for i in range(len(data)):
        headRot = math.Quat(data.at[i, head_rx], data.at[i, head_ry], data.at[i, head_rz], data.at[i, head_rw])

        vector = math.Vec3(data.at[i, vx], data.at[i, vy], data.at[i, vz])

        # compute head rotation with pitch and roll removed
        swingRot, twistRot = headRot.swingTwistDecomposition(math.Vec3(0, 1, 0))
        invHeadRot = twistRot.inverse()
        relVector = invHeadRot.rotate(vector)

        data.at[i, vx] = relVector.x
        data.at[i, vy] = relVector.y
        data.at[i, vz] = relVector.z
