## Image IO
def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed. 

    Parameters
    ----------
    filename: string
    color: boolean
      flag for color format. True (default) loads as RGB while False
      loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or 
        of size (H x W x 1) in grayscale. 
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not
        color)).astype(np.float32) # uint8(0-255) -> float64 -> float32(0-1)
    if img.ndim == 2:
        img = img[:, :, np.newaxis] # H x W x 1
        if color:
            img = np.tile(img, (1, 1, 3)) # H x W x 3
    elif img.shape[2] == 4:
        img = img[:, :, :3] # H x W x 3
    return img 


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims: (height, width) tuple of new dimensions
    interp_order: interpolation order, default is linear.

    Returns
    -------
    im: resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order,
                    mode='constant')
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty()


def oversample(images, crop_dims):
    """

    """



class Transformer:
    """
    Transform input for feeding into a Net.

    Note: this is mostly for illustrative purposes and it is likely better
    to define your own input preprocessing routine for your needs.

    Parameters
    ----------
    net: a Net for which the input should be prepared
    """

    def __init__(self, inputs):
        self.inputs = inputs
        self.transpose = {}
        self.channel_swap = {}
        self.raw_scale = {}
        self.mean = {}
        self.input_scale = {}

    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception('{} is not one of the net inputs: {}'.foramt(
                in_, self.inputs))

    def preprocess(self, in_, data):
        """
        Format input for Caffe:
        - convert to single
        - resize to input dimensions(preserving number of channels)
        - transpose dimensions to K x H x W
        - reorder channels (for instance color to BGR)
        - scale raw input (e.g. from [0, 1] to [0, 255] for ImageNet models)
        - subtract mean
        - scale feature

        Parameters 
        ----------
        in_ : name of input blob to preprocess for
        data : (H' x W' x K) ndarray

        Returns
        -------
        caffe_in : (K x H x W) ndarray for input to a Net
        """
        self.__check_input(in_)
        in_dims = self.inputs[in_][2:] # (height, width)
        transpose = self.transpose.get(in_)
        channel_swap = self.channel_swap.get(in_)
        raw_scale = self.raw_scale.get(in_)
        mean = self.mean.get(in_)
        input_scale = self.input_scale.get(in_)
        # conver to single(float 32)
        caffe_in = data.astype(np.float32, copy=False)
        # resize operation
        if caffe_in.shape[:2] != in_dims:
            caffe_in = resize_image(caffe_in, in_dims)
        # transpose operation
        if transpose is not None:
            caffe_in = caffe_in.transpose(transpose)
        # reorder operation
        if channel_swap is not None:
            caffe_in = caffe_in[channel_swap, :, :]
        # raw scale operation
        if raw_scale is not None:
            caffe_in *= raw_scale
        # subtract mean
        if mean is not None:
            caffe_in -= mean
        # multiply 1/standard deviation 
        if input_scale is not None:
            caffe_in *= input_scale 
    
    def 
