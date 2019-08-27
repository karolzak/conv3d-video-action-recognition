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
    """
    Creates and executes mPyPl pipe to load all videos from 'data_dir' (each subfolder is a separate class),
    extracts all frames and saves them in numpy format
    
    Parameters
    ----------
    data_dir : str, required
        The directory where all the vidoes are organised in subfolders (subfolder name=class name)
    ext : str, optional
        Extension of video files to search for, by default '.avi'
    target_ext : str, optional
        Target extension for video frames to be serialized to, by default '.allframes.npy'
    classes : dict, optional
        Dictionary with class names and numeral representations. Must match the folder names in 'data_dir'.
        If set to 'None' it will automatically figure out classes based on folders structure in 'data_dir'.
        Example {'Class1': 1, 'Class2': 2}
        Defaults to 'None'
    min_size : int, optional
        Minimum size of frames based on the shorter edge, by default 128
    max_elements : int, optional
        Max elements for silly progress indicator, by default 13320
    """
    (mp.get_datastream(data_dir, classes=classes, ext=ext)
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
    model,
    ext='.allframes.npy', 
    target_ext='.proc.c3d-avg.npy',
    classes=None, 
    frames_per_clip=16, 
    frames_step=8, 
    batch_size=32, 
    max_elements=13320):
    """
    Creates and executes mPyPl pipe to load all video frames, resize and crop them, preprocess,
    run inferencing against Keras model and serialize the resulting feature vectors as npy format
    
    Parameters
    ----------
    data_dir : str, required
        The directory where all the vidoes are organised in subfolders (subfolder name=class name)
    mean_std : array, required
        Array of per channel mean and std values used for preprocessing of frames.
        Template: array[ [mean_R, mean_G, mean_B], [std_R, std_G, std_B] ]
        Example: array[ [123, 112, 145], [60, 62, 64] ]
    model : Keras model obj, required
        Keras model object ready for running predictions
    ext : str, optional
        Extension of frames files to search for, by default '.allframes.npy'
    target_ext : str, optional
        Target extension for feature vectors to be serialized to, by default '.proc.c3d-avg.npy'
    classes : dict, optional
        Dictionary with class names and numeral representations. Must match the folder names in 'data_dir'.
        If set to 'None' it will automatically figure out classes based on folders structure in 'data_dir'.
        Example: {'Class1': 1, 'Class2': 2}
        Defaults to 'None'
    frames_per_clip : int, optional
        When extracting smaller clips from longer video this defines the number of frames cut out from longer clip, by default 16
    frames_step : int, optional
        When extracting smaller clips from longer video this defines the step in number of frames, by default 8
    batch_size : int, optional
        Mini batch size used when pushing data to the model for scoring, by default 32
    max_elements : int, optional
        Max elements for silly progress indicator, by default 13320
    """

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
    """
    Creates and executes mPyPl pipe to load feature vectors from serialized files and returns a preprocessed
    data stream that can be further used with respect to train/test split and specific classes assigned to each element in the stream
    
    Parameters
    ----------
    data_dir : str, required
        The directory where all the vidoes are organised in subfolders (subfolder name=class name)
    video_ext : str, optional
        Extension of video files to look for in 'data_dir', by default '.avi'
    features_ext : str, optional
        Extension of serialized feature vectors, by default '.proc.c3d-avg.npy'
    test_split : list, optional
        List of filenames belonging to the test subset. 
        If empty then there will be no data in the test subset, by default []
    classes : dict, optional
        Dictionary with class names and numeral representations. Must match the folder names in 'data_dir'.
        If set to 'None' it will automatically figure out classes based on folders structure in 'data_dir'.
        Example: {'Class1': 1, 'Class2': 2}
        Defaults to 'None'
    max_elements : int, optional
        Max elements for silly progress indicator, by default 13320
    
    Returns
    -------
    list of mPyPl.mdict.mdict
        List of dictionaries that can be used to access the data
    """

    data = (mp.get_datastream(data_dir, classes=classes, ext=video_ext)
        | mp.datasplit_by_pattern(test_pattern=test_split)
        | mp.pshuffle
        | mp.apply('filename', 'c3d_avg', 
                lambda fn: np.load(fn.replace(video_ext, features_ext)))
        | mp.silly_progress(elements=max_elements)
        | mp.select_fields(['c3d_avg', 'class_id', 'split'])
        | mp.as_list)
    return data