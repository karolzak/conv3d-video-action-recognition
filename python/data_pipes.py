import mPyPl as mp
from pipe import Pipe
from moviepy.editor import vfx, VideoFileClip
import numpy as np
from data_prep import close_clip, extract_clips, crop_center, preprocess_input, predict_c3d
from mpypl_pipe_func import cachecomputex


def videos_to_frames_pipe(
    data_dir, 
    ext='.avi', 
    target_ext='.allframes.npy', 
    classes=None, 
    min_size=128, 
    max_elements=13320):
    # process videos and save all frames as numpy arrays
    (mp.get_datastream(data_dir, classes=classes, ext=ext)
        | mp.take(20)
        | mp.apply('filename', 'clip',
                lambda fn: VideoFileClip(fn), 
                eval_strategy=mp.EvalStrategies.Value)
        | mp.apply('clip', 'clip', 
                lambda clip: clip.fx(vfx.resize, width=min_size) if clip.w <= clip.h else clip.fx(vfx.resize, height=min_size), 
                eval_strategy=mp.EvalStrategies.Value)
        | mp.apply('clip', 'allframes',
                lambda c: np.asarray(list(c.iter_frames())), 
                eval_strategy=mp.EvalStrategies.Value)
        | mp.iter('clip', close_clip)
        | mp.delfield('clip')       
        | cachecomputex(ext, target_ext,
                        lambda x,nfn: np.save(nfn, x['allframes']),
                        lambda x,nfn: print("Skipping saving 'allframes' for {}".format(x['filename'])))
        | mp.silly_progress(elements=max_elements) 
        | mp.execute)


def frames_to_features_pipe(
    data_dir,
    mean_std,
    ext='.allframes.npy', 
    target_ext='.proc.c3d-avg.npy',
    model=None,
    classes=None, 
    frames_per_clip=16, 
    frames_step=8, 
    batch_size=32, 
    max_elements=13320):

    (mp.get_datastream(data_dir, classes=classes, ext=ext)
        | mp.apply('filename', 'allframes',
                lambda fn: np.load(fn), 
                eval_strategy=mp.EvalStrategies.OnDemand) 
        | mp.apply('allframes', 'clips16-8', 
                lambda v: extract_clips(v, frames_per_clip=frames_per_clip, step=frames_step), 
                eval_strategy=mp.EvalStrategies.OnDemand)   
        | mp.apply('clips16-8', 'cropped16-8', 
                lambda v: np.asarray([[crop_center(frame) for frame in clip] for clip in v]), 
                eval_strategy=mp.EvalStrategies.OnDemand)   
        | mp.apply('cropped16-8', 'proc_cropped16-8', 
                lambda v: preprocess_input(v, mean_std, divide_std=False), 
                eval_strategy=mp.EvalStrategies.OnDemand) 
        | mp.apply_batch('proc_cropped16-8', 'c3d16-8',
                        lambda x: predict_c3d(x, model),
                        batch_size=batch_size)     
        | mp.apply('c3d16-8', 'c3d_avg', 
                lambda v: np.average(v, axis=0), 
                eval_strategy=mp.EvalStrategies.OnDemand) 
        | mp.silly_progress(elements=max_elements)
        | cachecomputex(ext, target_ext,
                        lambda x,nfn: np.save(nfn, x['c3d_avg']),
                        lambda x,nfn: print("Skipping saving 'c3d_avg' {}".format(x['filename'])))      
        | mp.execute)


def get_features_from_files(
    data_dir, 
    video_ext='.avi',
    features_ext='.proc.c3d-avg.npy', 
    test_split=[], 
    classes=None, 
    max_elements=13320):

    data = (mp.get_datastream(data_dir, classes=classes, ext=video_ext)
        | mp.datasplit_by_pattern(test_pattern=test_split)
        | mp.pshuffle
        | mp.apply('filename', 'c3d_avg', 
                lambda fn: np.load(fn.replace(video_ext, features_ext)))
        | mp.silly_progress(elements=max_elements)
        | mp.select_fields(['c3d_avg', 'class_id', 'split'])
        | mp.as_list)
    return data