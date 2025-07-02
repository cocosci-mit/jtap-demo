import cv2
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd

def process_video_stimulus(rgb_video_original, skip_t = 1):

    onehot_video = []

    for i, img in enumerate(rgb_video_original):

        if i % skip_t != 0:
            continue

        blue_mask = (img[..., 0] < 100) & \
                    ((img[..., 1]) < 100) & \
                    ((img[..., 2]) > 220)

        original_frame_onehot = np.zeros(tuple(img.shape[:2]), dtype=np.uint8)
        original_frame_onehot[blue_mask] = 2

        red_mask = (img[..., 0] > 220) & \
                    ((img[..., 1]) < 100) & \
                    ((img[..., 2]) < 100)
        original_frame_onehot[red_mask] = 4

        green_mask = (img[..., 0] < 100) & \
                    ((img[..., 1]) > 220) & \
                    ((img[..., 2]) < 100)
        original_frame_onehot[green_mask] = 5

        black_mask = (img[..., 0] < 50) & \
                    ((img[..., 1]) < 50) & \
                    ((img[..., 2]) < 50)
        original_frame_onehot[black_mask] = 3

        gray_mask = (img[..., 0] < 138) & \
                    (img[..., 0] > 118) & \
                    (img[..., 1] < 138) & \
                    (img[..., 1] > 118) & \
                    (img[..., 2] < 138) & \
                    (img[..., 2] > 118)
        original_frame_onehot[gray_mask] = 1


        onehot_video.append(original_frame_onehot)

    return np.array(onehot_video).astype(np.int8)

def load_stimuli(trial_number, trial_folder, skip_t = 2, get_high_res = False):
    trial_path = os.path.join(trial_folder, trial_number, 'low_res_obs.npz')
    if get_high_res:
        trial_path = os.path.join(trial_folder, trial_number, 'high_res_obs.npz')
        high_res_video = np.load(trial_path)['arr_0']
    rgb_video = np.load(trial_path)['arr_0']
    onehot_video = process_video_stimulus(rgb_video, skip_t = skip_t)
    onehot_video_obs = jnp.flip(jnp.transpose(onehot_video, (0, 2, 1)), axis=2)
    onehot_video_obs_np = np.array(onehot_video_obs)
    if get_high_res:
        return rgb_video, onehot_video_obs, onehot_video_obs_np, high_res_video
    return rgb_video, onehot_video_obs, onehot_video_obs_np

