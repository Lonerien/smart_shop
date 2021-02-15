# -*- coding: utf-8 -*-

import abc

            
class Detector(object, metaclass=abc.ABCMeta):
    '''
    Sklearn-like interface for object detectors.
    All detectors must be derived from this abstract class.
    '''
    @abc.abstractmethod
    def __init__():
        pass
        
    @abc.abstractmethod
    def fit():
        '''
        Train the model
        '''
        raise NotImplementedError('Implement fit() in {}'.format(self.__class__.__name__))

    @abc.abstractmethod
    def predict():
        '''
        Predict the bounding boxes for given image
        '''
        raise NotImplementedError('Implement predict() in {}'.format(self.__class__.__name__))
        