# mPyPl - Monadic Pipeline Library for Python
# http://github.com/shwars/mPyPl
# Fixed temp function to generate batches based on original function from mPyPl package
# TODO: Replace it with original function once pull request goes through

from pipe import Pipe
import numpy as np
import os


@Pipe
def cachecomputex(seq, orig_ext, new_ext, func_yes=None, func_no=None, filename_field='filename'):
    """
    Given a sequence with filenames in `filename_field` field with extension `orig_ext`, compute and save new files with extension `new_ext`.
    If target file does not exist, `func_yes` is executed (which should produce that file), otherwise `func_no`.
    Original sequence is returned. `func_yes` and `func_no` are passed the original `mdict` and new file name.
    """
    for x in seq:
        fn = x[filename_field]
        nfn = fn.replace(orig_ext,new_ext)
        if not os.path.isfile(nfn):
            if func_yes: func_yes(x,nfn)
        else:
            if func_no: func_no(x,nfn)
        yield x


@Pipe
def as_batch2(flow, feature_field_name='features', label_field_name='label', batchsize=16, out_features_dtype=None, out_labels_dtype=None):
    """
    Split input datastream into a sequence of batches suitable for keras training.
    :param flow: input datastream
    :param feature_field_name: feature field name to use. can be string or list of strings (for multiple arguments). Defaults to `features`
    :param label_field_name: Label field name. Defaults to `label`
    :param batchsize: batch size. Defaults to 16.
    :return: sequence of batches that can be passed to `flow_generator` or similar function in keras
    """
    #TODO: Test this function on multiple inputs!
    batch = labels = None
    while (True):
        for i in range(batchsize):
            data = next(flow)
            # explicitly compute all fields - this is needed for all fields to be computed only once for on-demand evaluation
            flds = { i : data[i] for i in (feature_field_name if isinstance(feature_field_name, list) else [feature_field_name])}
            lbls = data[label_field_name] # TODO: what happens when label_field_name is a list?
                
            if batch is None:
                if isinstance(feature_field_name, list):
                    batch = [np.zeros((batchsize,)+flds[i].shape, dtype=flds[i].dtype if out_features_dtype is None else out_features_dtype) for i in feature_field_name]
                else:
                    batch = np.zeros((batchsize,)+flds[feature_field_name].shape, dtype=flds[feature_field_name].dtype if out_features_dtype is None else out_features_dtype)
                    
                lbls_shape = lbls.shape if type(lbls) is np.ndarray else (1,)
                out_labels_dtype = out_labels_dtype if out_labels_dtype is not None else lbls.dtype if type(lbls) is np.ndarray else None
                labels = np.zeros((batchsize,)+lbls_shape, dtype=out_labels_dtype)
            if isinstance(feature_field_name, list):
                for j,n in enumerate(feature_field_name):
                    batch[j][i] = flds[n]
            else:
                batch[i] = flds[feature_field_name]
            labels[i] = lbls
        yield (batch, labels)
        batch = labels = None
