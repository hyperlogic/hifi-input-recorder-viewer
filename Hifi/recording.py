# Hifi input recording loader
import math
import json
import gzip
import pandas
import numpy
import scipy.signal
from . import vecmath

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
    prev_q = vecmath.Quat(0.0, 0.0, 0.0, 1.0)
    for i in range(len(data)):
        q = vecmath.Quat(data.at[i, x_key], data.at[i, y_key], data.at[i, z_key], data.at[i, w_key])
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
    prev_rot = vecmath.Quat(0.0, 0.0, 0.0, 1.0)
    for i in range(len(data)):
        q = vecmath.Quat(data.at[i, x_key], data.at[i, y_key], data.at[i, z_key], data.at[i, w_key])
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

        xform_pos = vecmath.Vec3(data.at[i, xform_px], data.at[i, xform_py], data.at[i, xform_pz])
        xform_rot = vecmath.Quat(data.at[i, xform_rx], data.at[i, xform_ry], data.at[i, xform_rz], data.at[i, xform_rw])
        xform = vecmath.Xform(xform_pos, xform_rot)

        pos = vecmath.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])
        rot = vecmath.Quat(data.at[i, rx], data.at[i, ry], data.at[i, rz], data.at[i, rw])

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

        xform_pos = vecmath.Vec3(data.at[i, xform_px], data.at[i, xform_py], data.at[i, xform_pz])
        xform_rot = vecmath.Quat(data.at[i, xform_rx], data.at[i, xform_ry], data.at[i, xform_rz], data.at[i, xform_rw])
        xform = vecmath.Xform(xform_pos, xform_rot)
        xform = xform.inverse()

        pos = vecmath.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])
        rot = vecmath.Quat(data.at[i, rx], data.at[i, ry], data.at[i, rz], data.at[i, rw])

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
        xform_rot = vecmath.Quat(data.at[i, xform_rx], data.at[i, xform_ry], data.at[i, xform_rz], data.at[i, xform_rw])

        pos = vecmath.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])

        newPos = xform_rot.rotate(pos)

        data.at[i, px] = newPos.x
        data.at[i, py] = newPos.y
        data.at[i, pz] = newPos.z

def apply_xform_inverse_vector(data, xform_rx, xform_ry, xform_rz, xform_rw, px, py, pz):
    """Mutates the position columns passed in from a pandas.DataFrame by the inverse of the given xform"""
    for i in range(len(data)):
        xform_rot = vecmath.Quat(data.at[i, xform_rx], data.at[i, xform_ry], data.at[i, xform_rz], data.at[i, xform_rw])
        xform_rot = xform_rot.inverse()

        pos = vecmath.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])

        newPos = xform_rot.rotate(pos)

        data.at[i, px] = newPos.x
        data.at[i, py] = newPos.y
        data.at[i, pz] = newPos.z

def convert_to_horizontal_relative_xform(data, head_px, head_py, head_pz,
                                         head_rx, head_ry, head_rz, head_rw,
                                         px, py, pz, rx, ry, rz, rw):
    """Mutates the position and rotaiton columns passed in from a pandas.DataFrame to relative to the horizontal head"""

    for i in range(len(data)):
        headPos = vecmath.Vec3(data.at[i, head_px], data.at[i, head_py], data.at[i, head_pz])
        headRot = vecmath.Quat(data.at[i, head_rx], data.at[i, head_ry], data.at[i, head_rz], data.at[i, head_rw])

        pos = vecmath.Vec3(data.at[i, px], data.at[i, py], data.at[i, pz])
        rot = vecmath.Quat(data.at[i, rx], data.at[i, ry], data.at[i, rz], data.at[i, rw])

        # compute head rotation with pitch and roll removed
        swingRot, twistRot = headRot.swingTwistDecomposition(vecmath.Vec3(0, 1, 0))
        invHead = vecmath.Xform(headPos, twistRot)
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


def convert_to_horizontal_relative_vector(data, head_px, head_py, head_pz,
                                          head_rx, head_ry, head_rz, head_rw,
                                          vx, vy, vz):
    """Mutates the vector columns passed in from a pandas.DataFrame to relative to the horizontal head"""

    for i in range(len(data)):
        headRot = vecmath.Quat(data.at[i, head_rx], data.at[i, head_ry], data.at[i, head_rz], data.at[i, head_rw])

        vector = vecmath.Vec3(data.at[i, vx], data.at[i, vy], data.at[i, vz])

        # compute head rotation with pitch and roll removed
        swingRot, twistRot = headRot.swingTwistDecomposition(vecmath.Vec3(0, 1, 0))
        invHeadRot = twistRot.inverse()
        relVector = invHeadRot.rotate(vector)

        data.at[i, vx] = relVector.x
        data.at[i, vy] = relVector.y
        data.at[i, vz] = relVector.z

def nan_check(data):
    for key in data.keys():
        if data[key].isnull().values.any():
            print("WARNING: data['{}'] has {} NaNs it!".format(key, data[key].isnull().sum()))

def prepare_data_for_visualization(input_recording_filename, trajectory_frames):
    POSE_NAMES = ['Head', 'Hips', 'LeftFoot', 'RightFoot', 'LeftHand', 'RightHand']
    SUB_KEYS = {'rotation': ['rx', 'ry', 'rz', 'rw'],
                'translation': ['px', 'py', 'pz']}

    print("Loading recording", input_recording_filename)
    data = load(input_recording_filename, POSE_NAMES, SUB_KEYS)

    nan_check(data)

    #
    # Transform all poses into sensor space, if possible
    #

    if 'sensor_px' in data:
        # transform data from avatar to sensor space.
        for pose in POSE_NAMES:
            print("transforming " + pose + " into sensor space")
            apply_xform(data, 1.0, 'avatar_px', 'avatar_py', 'avatar_pz', 'avatar_rx', 'avatar_ry', 'avatar_rz', 'avatar_rw',
                                       pose + '_px', pose + '_py', pose + '_pz', pose + '_rx', pose + '_ry', pose + '_rz', pose + '_rw')
            apply_xform_inverse(data, 'sensor_s', 'sensor_px', 'sensor_py', 'sensor_pz', 'sensor_rx', 'sensor_ry', 'sensor_rz', 'sensor_rw',
                                               pose + '_px', pose + '_py', pose + '_pz', pose + '_rx', pose + '_ry', pose + '_rz', pose + '_rw')
            nan_check(data)
    else:
        print("Recording is in avatar space")

    #
    # compute motion of root by filtering the motion of the hips.
    #

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

    print("filtering hips to generate root motion")

    # Filter settings
    filter_order = 3
    filter_fs = 90.0  # sample rate, Hz
    filter_cutoff = 0.5  # desired cutoff frequency of the filter, Hz

    root_y = data["Hips_py"].mean()
    num_samples = len(data["Hips_py"])
    data["Root_px"] = butter_lowpass_filter(data["Hips_px"], filter_cutoff, filter_fs, filter_order)
    data["Root_py"] = pandas.Series([root_y for i in range(num_samples)])
    data["Root_pz"] = butter_lowpass_filter(data["Hips_pz"], filter_cutoff, filter_fs, filter_order)

    #
    # compute rotation of root by filtering rotation of the hips
    #

    thetas = []
    z_axis = vecmath.Vec3(0, 0, 1)
    y_axis = vecmath.Vec3(0, 1, 0)
    for i in range(num_samples):
        rot = vecmath.Quat(data["Hips_rx"][i], data["Hips_ry"][i], data["Hips_rz"][i], data["Hips_rw"][i])
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
    rot_quats = [vecmath.Quat.fromAngleAxis(theta, y_axis) for theta in filtered_thetas]

    data["Root_rx"] = pandas.Series([q.x for q in rot_quats])
    data["Root_ry"] = pandas.Series([q.y for q in rot_quats])
    data["Root_rz"] = pandas.Series([q.z for q in rot_quats])
    data["Root_rw"] = pandas.Series([q.w for q in rot_quats])
    POSE_NAMES.append("Root")

    nan_check(data)

    #
    # compute trajectories of root motion
    #

    def timeShift(series, offset):
        num_samples = len(series)
        return [series[max(0, min(num_samples - 1, i + offset))] for i in range(num_samples)]

    num_trajectory_points = len(trajectory_frames)
    suffixes = ["_px", "_py", "_pz", "_rx", "_ry", "_rz", "_rw"]
    for i in range(num_trajectory_points):
        for suffix in suffixes:
            data["RootTraj{}{}".format(i + 1, suffix)] = timeShift(data["Root{}".format(suffix)], trajectory_frames[i])

    #
    # compute trajectories of head
    #

    num_trajectory_points = len(trajectory_frames)
    suffixes = ["_px", "_py", "_pz", "_rx", "_ry", "_rz", "_rw"]
    for i in range(num_trajectory_points):
        for suffix in suffixes:
            data["HeadTraj{}{}".format(i + 1, suffix)] = timeShift(data["Head{}".format(suffix)], trajectory_frames[i])

    return data



def prepare_data_for_motion_matching(input_recording_filename, trajectory_frames):

    data = prepare_data_for_visualization(input_recording_filename, trajectory_frames)
    num_trajectory_points = len(trajectory_frames)

    #
    # Transform everything back into Root relative space
    #

    output_poses = ['Hips', 'LeftFoot', 'RightFoot', 'Head', 'LeftHand', 'RightHand']
    for i in range(num_trajectory_points):
        output_poses.append("RootTraj{}".format(i + 1))
    for i in range(num_trajectory_points):
        output_poses.append("HeadTraj{}".format(i + 1))

    for pose in output_poses:
        print("transforming " + pose + " into Root relative space")

        apply_xform_inverse(data, 1.0, 'Root_px', 'Root_py', 'Root_pz', 'Root_rx', 'Root_ry', 'Root_rz', 'Root_rw',
                            pose + '_px', pose + '_py', pose + '_pz', pose + '_rx', pose + '_ry', pose + '_rz', pose + '_rw')

    #
    # Transform head trajectory into horizontal head relative space
    #

    num_trajectory_points = len(trajectory_frames)
    for i in range(num_trajectory_points):
        traj = "HeadTraj{}".format(i + 1)
        convert_to_horizontal_relative_xform(data, 'Head_px', 'Head_py', 'Head_pz', 'Head_rx', 'Head_ry', 'Head_rz', 'Head_rw',
                                             traj + '_px', traj + '_py', traj + '_pz', traj + '_rx', traj + '_ry', traj + '_rz', traj + '_rw')

    return data


def output_data_for_motion_matching(output_keys, data):
    #
    # output json for motion matching code
    #

    filename = 'data.json'
    print("Writing {}".format(filename))

    newData = pandas.DataFrame()
    for key in output_keys:
        newData[key] = data[key]

    with open(filename, 'w') as outfile:
        json.dump(newData.values.tolist(), outfile, indent = 4)

    #
    # Print c++ enum to ensure that json and C++ have the same indices.
    #

    print("enum RowIndices {")
    for i in range(len(output_keys)):
        key = output_keys[i]
        if i == 0:
            print("    {}_INDEX = 0,".format(key.upper()))
        else:
            print("    {}_INDEX,".format(key.upper()))

    print("    DATA_ROW_SIZE")
    print("};")

