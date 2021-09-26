from tensorflow.python.keras.models import Model


def create_hooknet_encoder(hooknet: Model):
    """Creates a new Keras model with bottleneck of the target branch as ouput after combining the context features

    Args:
        hooknet (Model): HookNet Model

    Returns:
        Model: return new model with encoding as output
    """
    encoding_layer = hooknet.get_layer("target-branchbottle").output
    return Model(hooknet.inputs, encoding_layer)


# This constraint assumes valid convolutions with stride of 1, 2x2 pooling and 2x2 upsampling
def check_input(depth, input_size, filter_size, n_convs):
    """checks if input is valid for model configuration 

    Args:
        depth (int): depth of the model
        input_size (int): shape of model (width or height)
        filter_size (int): filter size  convolutions
        n_convs (int): number of convolutions per depth
    """
    def is_even(size):
        return size % 2 == 0

    i1 = input_size
    # encoding
    for _ in range(depth):
        # input_size reduced through valid convs
        i1 -= (filter_size - 1) * n_convs
        # check if inputsize before pooling is even
        if not is_even(i1):
            return False

        # max pooling
        i1 /= 2
        if i1 <= 0:
            return False

    # decoding
    for _ in range(depth):
        # input_size reduced through valid convs
        i1 -= (filter_size - 1) * n_convs

        # check if inputsize before upsampling is even
        if not is_even(i1):
            return False

        # upsampling
        i1 *= 2
        i1 -= filter_size - 1

        if i1 <= 0:
            return False

    # check if inputsize is even
    if not is_even(i1):
        False

    i1_end = i1 - (filter_size - 1) * n_convs

    if i1_end <= 0:
        return False

    return True