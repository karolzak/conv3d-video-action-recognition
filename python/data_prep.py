import random
import os
from moviepy.editor import *
import numpy as np

def random_frames(video, target_num_frames):
    num_frames = video.shape[0]
    frames_idxs = random.sample(range(0, num_frames), target_num_frames)
    frames_idxs.sort()
    return video[frames_idxs]


def random_crop(img, mask=None, width=112, height=112, x=None, y=None):
    #assert img.shape[0] >= height
    #assert img.shape[1] >= width
    if x is None:
        x = random.randint(0, img.shape[1] - width)
    if y is None:
        y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    if mask is None:
        return img
    else:
        #assert img.shape[0] == mask.shape[0]
        #assert img.shape[1] == mask.shape[1]
        mask = mask[y:y+height, x:x+width]
        return img, mask


def extract_clips(video, frames_per_clip=16):
    total_frames = video.shape[0]
    num_clips = total_frames // frames_per_clip
    extracted_clips = [np.asarray(video[i:i+frames_per_clip]) for i in range(0,frames_per_clip*num_clips,frames_per_clip)]
    
    return np.asarray(extracted_clips, dtype=video.dtype)

