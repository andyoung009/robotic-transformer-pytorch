# Copied from kaggle bcz-dataset-stats 'https://www.kaggle.com/code/mjchang07/bcz-dataset-stats'
import glob
import tensorflow as tf
import itertools
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from IPython.display import Video
# from imageio import mimwrite
# !ffmpeg -codecs | grep VP9
from IPython.display import HTML
from base64 import b64encode
import cv2
import ffmpeg

IMG_KEY = 'present/image/encoded'
example_eids = ['1617320692734250831','1620429448629678610','1616524273044475002','1615832193510114888']
# 1615832193510114874
num_waypoints = 10
features = {
    # Current State & Goal.
    IMG_KEY: tf.io.FixedLenFeature([], tf.string),
    'episode_id': tf.io.FixedLenFeature([1], tf.string),
    'subtask_id': tf.io.FixedLenFeature([1], tf.int64),
    'subtask_name': tf.io.FixedLenFeature([1], tf.string),
    'sentence_embedding': tf.io.FixedLenFeature([512], tf.float32),
    'sequence_length': tf.io.FixedLenFeature([1], tf.int64),
    'present/xyz': tf.io.FixedLenFeature([3], tf.float32),
    'present/axis_angle': tf.io.FixedLenFeature([3], tf.float32),
    'present/sensed_close': tf.io.FixedLenFeature([1], tf.float32),
    # Episodes (e.g. video of robot performing task) are stored as contiguous Examples each
    # containing single image observations. The present/timestep_count contains the
    # timestep of the episode, which can be used to determine when an episode starts/ends.
    'present/timestep_count': tf.io.FixedLenFeature([1], tf.float32),
    # You may want to condition the model on whether a human was performing the demo
    # and whether a human is intervening over the policy.
    'human_executing': tf.io.FixedLenFeature([1], tf.int64),
    'present/intervention': tf.io.FixedLenFeature([1], tf.int64),
     # Future actions to predict.
    'future/xyz_residual': tf.io.FixedLenFeature([3*num_waypoints], tf.float32),
    'future/axis_angle_residual': tf.io.FixedLenFeature([3*num_waypoints], tf.float32),
    'future/target_close': tf.io.FixedLenFeature([1*num_waypoints], tf.float32),
    'present/target_close': tf.io.FixedLenFeature([1], tf.float32),
    'present/camera_pose_base': tf.io.FixedLenFeature([12], tf.float32),
    'present/camera_rgb/intrinsics': tf.io.FixedLenFeature([3,3], tf.float32)
}

def parse_tfrecord(serialized_example):
    example = tf.io.parse_single_example(serialized_example, features)
    example[IMG_KEY] = tf.io.decode_jpeg(example[IMG_KEY])
    return example

def play(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=1000 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)

def render(frames):
    size = np.flip(frames[0].shape[:2])
    out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for img in frames:
        img = np.flip(img,axis=-1)
        out.write(img)
    out.release()
    # !ffmpeg -hide_banner -loglevel error -y -i output_video.mp4 display.mp4 > /dev/null
    os.system('ffmpeg -hide_banner -loglevel error -y -i output_video.mp4 display.mp4 > /dev/null')
    # (
    # ffmpeg
    # .input('output_video.mp4')
    # .output('display.mp4', loglevel='error', overwrite_output=True)
    # .run(capture_stdout=True, capture_stderr=True)
    # )
# success21_filepattern = '/kaggle/input/bc-z-robot/bc-z-robot/bcz-21task_v9.0.1.tfrecord/bcz-21task_v9.0.1.tfrecord/train-*'
# ROOT = '/kaggle/input/bc-z-robot/bc-z-robot'
ROOT = '/LOG/realman/LLM/datasets/bczrobot/bc-z-robot'
source = ['bcz-79task_v16.0.0.tfrecord/bcz-79task_v16.0.0.tfrecord/val-*', 'bcz-79task_v16.0.0.tfrecord/bcz-79task_v16.0.0.tfrecord/train-*', 'bcz-21task_v9.0.1.tfrecord/bcz-21task_v9.0.1.tfrecord/train-*', 'bcz-21task_v9.0.1.tfrecord/bcz-21task_v9.0.1.tfrecord/val-*']
filenames = [glob.glob(os.path.join(ROOT,s)) for s in source]
print([len(x) for x in filenames])

ROOT_FAILURE = '/LOG/realman/LLM/datasets/bczrobot'
source = [ 'bcz-21task_v9.0.1_failures.tfrecord/bcz-21task_v9.0.1_failures.tfrecord/val*','bcz-21task_v9.0.1_failures.tfrecord/bcz-21task_v9.0.1_failures.tfrecord/train*', 'bcz-79task_v16.0.0_failures.tfrecord/bcz-79task_v16.0.0_failures.tfrecord/train*','bcz-79task_v16.0.0_failures.tfrecord/bcz-79task_v16.0.0_failures.tfrecord/val*']
filenames_failure = [glob.glob(os.path.join(ROOT_FAILURE,s)) for s in source]
print([len(x) for x in filenames_failure])

filenames = sorted(list(itertools.chain(*(filenames+filenames_failure))))

# success21_filenames = glob.glob(success21_filepattern)
ds = tf.data.TFRecordDataset(filenames)
ds = ds.map(parse_tfrecord)
print(len(filenames))
print(filenames[:10])
# print(os.listdir('/LOG/realman/LLM/datasets/bczrobot/bc-z-robot/bcz-21task_v9.0.1.tfrecord/bcz-21task_v9.0.1.tfrecord'))

ds_iter = ds.as_numpy_iterator()
np_examples = next(ds_iter)
for key in features:
    print(f'{key}: shape={np_examples[key].shape}')

img = np_examples[IMG_KEY]
Image.fromarray(img)

cur_eid = None
episode_stats = defaultdict(lambda: {'duplicate': 0})
last_ep = None
intervention_values = []
he_values = []
prev_frame = None
for sample in ds_iter:
    eid = sample['episode_id'][0].decode('utf-8')
    frame_num = sample['present/timestep_count']
    # if the episode id has changed
    if eid != cur_eid:
        if len(episode_stats) % 1000 == 0:
            print("Episodes: ",len(episode_stats))
        # ending an episode
        if cur_eid is not None:
            # sanity check to ensure all intervention values are the same
            if not all(np.array(intervention_values) == intervention_values[0]):
                raise Exception(f'not all frames had the same intervention value')
            if not all(np.array(he_values) == he_values[0]):
                raise Exception(f'not all frames had the same human_executing value')
        intervention_values = []
        he_values = []
        cur_eid = eid
        if cur_eid in episode_stats:
            print('dup eid ',cur_eid, ' pos ', episode_stats[cur_eid]['order'])
            raise Exception(f'episode id appearing twice, not back to back')
        else:
            if eid in example_eids:
                episode_stats[cur_eid]['rgb'] = []
                print('Found example eid')
            episode_stats[cur_eid]['frames'] = set()
            # print('new ',cur_eid, sample['present/timestep_count'])
            episode_stats[cur_eid]['human_executing'] = sample['human_executing'].item()
            episode_stats[cur_eid]['intervention'] = sample['present/intervention'].item()
            episode_stats[cur_eid]['order'] = len(episode_stats)
    elif prev_frame is None or frame_num < prev_frame:
        # frame count as gone down but episode id hasn't changed. Episode is reapeated
        # in thie dataset (I've visualized some of these to verify that it is a duplicate)
        # assert that it is infact a duplicated frame id
        assert frame_num.item() in cur_frames
        # mark this episode as having been duplicated
        episode_stats[cur_eid]['duplicate'] += 1
    # which frames have been loaded for the current trajectory
    cur_frames = episode_stats[cur_eid]['frames']
    prev_frame = frame_num
    frame_num = sample['present/timestep_count'].item()
    intervention_values.append(sample['present/intervention'].item())
    he_values.append(sample['human_executing'].item())
    if episode_stats[cur_eid]['duplicate'] > 0:
        # sanity check that no new frames ever come in for an episode that we've counted
        # as duplicated
        assert frame_num in cur_frames
    else:
        if eid in example_eids:
            episode_stats[cur_eid]['rgb'].append(sample[IMG_KEY])
    episode_stats[cur_eid]['frames'] = cur_frames.union([frame_num])

print("total trajectories: ", len(episode_stats))
with_human = [e for e in episode_stats.values() if e['human_executing'] == 0]
print("with human_executing 0: ", len(with_human))
with_intervention = [e for e in episode_stats.values() if e['intervention'] == 1]
print("with intervention indicated: ", len(with_intervention))
no_intervention = [e for e in episode_stats.values() if e['intervention'] == 0]
print("without intervention indicated: ", len(no_intervention))
duplicate_counts = [e['duplicate'] for e in episode_stats.values()]
print("number of duplicates: ", sum(duplicate_counts))
print("total with dupplicates: ", len(episode_stats)+sum(duplicate_counts))

# render(episode_stats[example_eids[0]]['rgb'])
# print("Intervention label: ",episode_stats[example_eids[0]]['intervention'])
# Video('./display.mp4',embed=True)

render(episode_stats[example_eids[1]]['rgb'])
print("Intervention label: ",episode_stats[example_eids[1]]['intervention'])
Video('./display1.mp4',embed=True)

# render(episode_stats[example_eids[2]]['rgb'])
# print("Intervention label: ",episode_stats[example_eids[2]]['intervention'])
# Video('./display2.mp4',embed=True)

# render(episode_stats[example_eids[3]]['rgb'])
# print("Intervention label: ",episode_stats[example_eids[3]]['intervention'])
# Video('./display3.mp4',embed=True)
