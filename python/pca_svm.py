from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import numpy as np


def run_pca(X_train, X_test, n_components):
    """
    Apply PCA on features
    
    Parameters
    ----------
    X_train : array
        Array of train features vector
    X_test : array
        Array of test features vector
    n_components : int
        Target dimension of features vector
    
    Returns
    -------
    array, array
        Returns train and test array of features vectors after applying PCA transformation
    """
    pca = PCA(n_components=n_components) 
    X_train_pca = pca.fit_transform(X_train) 
    X_test_pca = pca.transform(X_test)
    print("Train/test shapes after PCA: %s / %s" % (X_train_pca.shape, X_test_pca.shape))   
    return X_train_pca, X_test_pca


def run_svm(X_train, Y_train, X_test, Y_test, labels=None):
    """
    Trains a simple linear SVM and prints out classification report
    
    Parameters
    ----------
    X_train : array
        Array of train features vector
    Y_train : array
        Array of train labels
    X_test : array
        Array of test features vector
    Y_test : array
        Array of test labels
    labels : list, optional
        List of label names used when printing classification report, by default None
    
    Returns
    -------
    array
        Array of predicted labels
    """
    clf = LinearSVC(random_state=0)
    clf.fit(X_train, Y_train)  
    
    Y_test_pred = clf.predict(X_test)
    count_correct = np.sum(Y_test == Y_test_pred)
    
    # Printing classification report
    print(classification_report(list(Y_test), list(Y_test_pred), target_names=labels))
    return Y_test_pred


def run_pca_svm(X_train, Y_train, X_test, Y_test, n_components, labels=None):
    """
    Complete function that runs through PCA followed by SVM training
    
    Parameters
    ----------
    X_train : array
        Array of train features vector
    Y_train : array
        Array of train labels
    X_test : array
        Array of test features vector
    Y_test : array
        Array of test labels
    n_components : int
        Target dimension of features vector
    labels : list, optional
        List of label names used when printing classification report, by default None
    
    Returns
    -------
    array
        Array of predicted labels
    """
    X_train_pca, X_test_pca = run_pca(X_train, X_test, n_components)
    return run_svm(X_train_pca, Y_train, X_test_pca, Y_test, labels=labels)    