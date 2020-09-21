# Author :  Jean-Baptiste PROST
# Date : Summer 2020
import numpy as np
from scipy.ndimage import shift, rotate

class PreProcessing:
    """
    The class manages the transformations of image/video samples.
    The creation of the instance will define which pre-processing will be applied.

    """
    def __init__(self, dim, pre_processing ):
        """
        Which pre-processing should be done
        """
        # - Dimension of the data : changes the way files are managed
        if len(dim) != 4:
            raise ValueError(' The data must be of rank 4, {} ranked received'.format(len(dim)))
        else:
            self.dim = dim  # dimension of a sample [n_sequence, height, width, chamber view]
            if self.dim[-1] < 2:  # if the sample has only a single channel
                self.multi_channel = False
                self._preprocess_sample_ = self._preprocess_sample_single_
            else:
                self.multi_channel = True
                self._preprocess_sample_ = self._preprocess_sample_multiple_

        # default initialization
        self.normalize = False
        self.shift = 0
        self.rotate = 0

        if pre_processing is not None:
            if ('normalize' in pre_processing) and (pre_processing['normalize']):
                self.normalize = True

            if ('shift' in pre_processing) and (pre_processing['shift']):
                self.shift = pre_processing['shift'] + 1

            if ('rotate' in pre_processing) and (pre_processing['rotate']):
                self.rotate = pre_processing['rotate'] + 1

    # --- Dimensionality and Pre processing functions ---
    def _preprocess_sample_single_(self, x):
        """
        Returns the samples preprocessed with additional dimension
        """
        return self._preprocessing_(x)[..., np.newaxis]

    def _preprocess_sample_multi_(self, x):
        """
        Returns the samples preprocessed with appropriate rank
        """
        return self._preprocessing_(x)

    def _preprocessing_(self, array):
        """
        Applies a random pre processing to a single sample
        """
        if self.normalize:
            array = self._normalize_(array)
        if self.shift:
            array = self._shift_(array)
        if self.rotate:
            array = self._rotate_(array)
        return array

    # ---  Transformations ---
    def _normalize_(self, array):
        """
        Normalize each frame along the first dimension
        """
        if self.multi_channel:
            return array / array.max(axis=(1,2))[:,np.newaxis, np.newaxis,:]
        else:
            return array / array.max(axis=(1,2))[:,np.newaxis, np.newaxis]

    def _shift_(self, array):
        """
        Shift range. Use scipy method
        """
        shift_pix = np.random.randint(0, self.shift)
        if self.multi_channel:
            shifted = shift(array, [0, 0, shift_pix, 0])
        else:
            shifted = shift(array, [0, 0, shift_pix])
        return shifted

    def _rotate_(self, array):
        """
        Applies a random rotation of the image in the
        [ -'rotate_range' ; 'rotate_range' ] range to a stack
        """
        # get a random angle
        angle = np.random.randint(0, self.rotate)
        # get a random sign for the angle
        sign = np.random.randint(0, 2)
        return rotate(array, -sign * angle, (1, 2), reshape=False)
