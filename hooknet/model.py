from typing import Dict, List, Tuple
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Cropping2D,
    Input,
    MaxPooling2D,
    Multiply,
    Reshape,
    Subtract,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, Optimizer
from tensorflow.python.framework.ops import Tensor
from hooknet.utils import check_input


class HookNet(Model):

    """
    HookNet: a convolutional neural network with mulitple branches of encoder-decoders for the task of semantic segmenation.

    ...

    Attributes
    ----------

    input_shape: List[int]
        the input shape of the model

    output_shape: List[int]
        the output shape of the model before flattening.

    """

    def __init__(
        self,
        input_shape: List[int],
        hook_indices: tuple,
        n_classes: int,
        model_weights=None,
        depth: int = 4,
        n_convs: int = 2,
        filter_size: int = 3,
        n_filters: int = 16,
        padding: str = "valid",
        batch_norm: bool = False,
        activation: str = "relu",
        learning_rate: float = 0.00005,
        opt_name: str = "adam",
        l2_lambda: float = 0.001,
        loss_weights: List[float] = [1.0, 0.0],
        merge_type: str = "concat",
        predict_target_only: bool = True,
    ) -> None:

        """
        Parameters
        ----------
        input_shape : List[int]
            the input shape of the model for both branches

        n_classes: int
            the possible number of classes in the output of the model

        hook_indices: List[int]
            the respective depths (starting from 0) of hooking [from, to] in the decoders

        depth: int
            the depth of the encoder-decoder branches

        n_convs: int
            the number of 2D convolutions per convolutional block

        filter_size: int
            the size of the filter in a 2D convolution

        n_filters: intv
            the number of starting filters (will be increased and decreased by a factor 2 in each conv block in the encoders and decoders, respectively)

        padding: str
            padding type in 2D convolution (either 'same' or 'valid')

        batch_norm: bool
            boolean for using batch normalization

        activation: str
            activation function after 2D convolution

        learning_rate: float
            learning rate of the optimizer

        opt_name: str
            optimizer name (either 'sgd' or 'adam')

        l2_lambda: float
            l2 value for regulizer

        loss_weights: bool
            loss contribution for each branch

        merge_type: str
            method used for combining feature maps (either 'concat', 'add', 'subtract', 'multiply')
        """

        if input_shape[0] != input_shape[1]:
            raise ValueError("input shapes of both branches should be the same")

        if not check_input(
            depth=depth,
            input_size=input_shape[0][0],
            filter_size=filter_size,
            n_convs=n_convs,
        ):
            raise ValueError("input_shapes are not valid model parameters")

        self._input_shape_target = input_shape[0]
        self._input_shape_context = input_shape[1]

        self._n_classes = n_classes
        self._hook_indices = {(depth - 1) - hook_indices[0]: hook_indices[1] - 1}
        self._depth = depth
        self._n_convs = n_convs
        self._filter_size = filter_size
        self._n_filters = n_filters
        self._padding = padding
        self._batch_norm = batch_norm
        self._activation = activation
        self._learning_rate = learning_rate
        self._opt_name = opt_name
        self._l2_lambda = l2_lambda
        self._loss_weights = loss_weights
        self._merge_type = merge_type
        self._predict_target_only = predict_target_only
        # set l2 regulizer
        self._l2 = regularizers.l2(self._l2_lambda)

        # determine multi-loss model from loss weights
        self._multi_loss = any(loss_weights[1:])
        # construct model

        self._construct_hooknet()

        if model_weights is not None:
            print(f"loading weights... {model_weights}")
            self.load_weights(model_weights)

    @property
    def input_shape(self) -> List[int]:
        """Return the input shape of the model"""

        return self._input_shape_target

    @property
    def output_shape(self) -> List[int]:
        """Return the output shape of the model before flattening"""

        return self._out_shape

    @property
    def multi_loss(self) -> bool:
        return self._multi_loss

    def train_on_batch(
        self,
        x,
        y,
        sample_weight=None,
        class_weight=None,
        reset_metrics=True,
        return_dict=False,
    ):
        # flatten
        if self.multi_loss:
            y = y.reshape(y.shape[0], y.shape[1], y.shape[2] * y.shape[3], y.shape[4])
            y = list(y)
        else:
            y = y.reshape(y.shape[0], y.shape[1] * y.shape[2], y.shape[3])

        return super().train_on_batch(
            x,
            y=y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            reset_metrics=reset_metrics,
            return_dict=return_dict,
        )

    def predict_on_batch(self, x, reshape=True, argmax=True):
        if self.multi_loss and self._predict_target_only:
            predictions = np.array(super().predict_on_batch(x))[:, 0, :]
        else:
            predictions = super().predict_on_batch(x)

        if reshape:
            if self.multi_loss and not self._predict_target_only:
                predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], *self._out_shape)
            else:
                predictions = predictions.reshape(predictions.shape[0], *self._out_shape)

        if argmax:
            predictions = np.argmax(predictions, axis=-1) + 1
        return predictions

    def _construct_hooknet(self) -> None:
        """Construction of single/multi-loss model with multiple inputs and single/multiple outputs"""

        # declaration of context input
        input_2 = Input(self._input_shape_context)

        # construction of context branch and context hooks
        flatten2, context_hooks = self._construct_branch(
            input_2, reshape_name="reshape_context"
        )

        # declaration of target inpput
        input_1 = Input(self._input_shape_target)

        # construction of target branch with context hooks
        flatten1, _ = self._construct_branch(
            input_1, context_hooks, reshape_name="reshape_target"
        )

        # create single/multi loss model
        if self._multi_loss:
            self._create_model([input_1, input_2], [flatten1, flatten2])
        else:
            self._create_model([input_1, input_2], flatten1)

    def _construct_branch(
        self, input: Input, in_hooks: Dict = {}, reshape_name: str = "reshape_target"
    ) -> Tuple[Tensor, Dict]:
        """
        Construction of single branch

        Parameters
        ----------
        input : Input
            keras Input Tensor
        in_hooks : Dict
            A mapping for hooking from the context branch to the target branch
        reshape_name: str
            name for Reshape Tensor


        Returns
        -------
        flatten: Tensor
            last Tensor of the branch
        out_hooks: Dict
            mapping for hooking between branches
        """

        # input
        net = input
        net = Rescaling(scale=1.0 / 255.0, offset=0.0, name=None)(net)

        # encode and retreive residuals
        net, residuals = self._encode_path(net)

        # mid conv block
        net = self._conv_block(net, self._n_filters * 2 * (self._depth + 1))

        # decode and retreive hooks
        net, out_hooks = self._decode_path(net, residuals, in_hooks)

        # softmax output
        net = Conv2D(self._n_classes, 1, activation="softmax")(net)

        # set output shape
        self._out_shape = int_shape(net)[1:]

        # Reshape net
        flatten = Reshape(
            (self._out_shape[0] * self._out_shape[1], self._out_shape[2]),
            name=reshape_name,
        )(net)

        # return flatten output and hooks
        return flatten, out_hooks

    def _create_model(self, inputs: List[Input], outputs: List[Tensor]) -> None:
        """
        Creation of model

        Parameters
        ----------
        inputs: List[Input]
            inputs to the context and target branch
        output: List[Reshape]
            output(s) of the (context) and target branch

        """

        # initilization of keras model
        super().__init__(inputs, outputs)

        # set losses and loss weigths
        losses = (
            {
                "reshape_target": "categorical_crossentropy",
                "reshape_context": "categorical_crossentropy",
            }
            if self._multi_loss
            else {"reshape_target": "categorical_crossentropy"}
        )
        loss_weights = (
            {
                "reshape_target": self._loss_weights[0],
                "reshape_context": self._loss_weights[1],
            }
            if self._multi_loss
            else {"reshape_target": self._loss_weights[0]}
        )

        sample_weights_mode = (
            ["temporal", "temporal"] if self._multi_loss else "temporal"
        )

        # compile model
        self.compile(
            optimizer=self._opt(),
            loss=losses,
            sample_weight_mode=sample_weights_mode,
            loss_weights=loss_weights,
            weighted_metrics=["accuracy", CategoricalCrossentropy()],
        )

    def _opt(self) -> Optimizer:
        """
        Set optimizer

        Returns
        -------
        SGD or ADAM optimizer

        Raises
        ------

        ValueError: unsupported optimizer

        """

        # Set Gradient-descent optimizer
        if self._opt_name == "sgd":
            return SGD(lr=self._learning_rate)

        # Set Adam optimizer
        if self._opt_name == "adam":
            return Adam(lr=self._learning_rate)

        raise ValueError(f"unsupported optimizer name: {self._opt_name}")

    def _encode_path(self, net) -> Tuple[Tensor, List[Tensor]]:
        """
        Encoder

        Parameters
        ----------

        net: Tensor
            current Tensor in the model


        Returns
        -------
        net: Tensor
            current Tensor in the model
        residuals: List[Tensors]
            all the Tensors used residuals/skip connections in the decoder part of the model

        """

        # list for keeping track for residuals/skip connections
        residuals = []

        # set start filtersize
        n_filters = self._n_filters

        # loop through depths
        for b in range(self._depth):
            # apply convblock
            net = self._conv_block(net, n_filters)

            # keep Tensor for residual/sip connection
            residuals.append(net)

            # downsample
            net = self._downsample(net)

            # increase number of filters with factor 2
            n_filters *= 2

        return net, residuals

    def _decode_path(self, net: Tensor, residuals: List, inhooks: Dict = {}) -> Tensor:
        """
        Decoder

        Parameters
        ----------
        net: Tensor
            current Tensor in the model
        residuals: List[Tensors]
            all the Tensors used residuals/skip connections in the decoder part of the model
        in_hooks: Dict
            mapping for hooking between branches

        Returns
        -------
        net: Tensor
            current Tensor in the model
        hooks: Dict
            mapping between index and Tensor in model for hooking between branches

        """

        # list for keeping potential hook Tensors
        outhooks = []

        # set start number of filters of decoder
        n_filters = self._n_filters * 2 * self._depth

        # loop through depth in reverse
        for b in reversed(range(self._depth)):

            # hook if hook is available
            if b in inhooks:
                # combine feature maps via merge type
                if self._merge_type == "concat":
                    net = self._concatenator(
                        net, inhooks[b], name="target-branchbottle"
                    )
                else:
                    net = self._merger(net, inhooks[b])

            # upsample
            net = self._upsample(net, n_filters)

            # concatenate residuals/skip connections
            net = self._concatenator(net, residuals[b])

            # apply conv block
            net = self._conv_block(net, n_filters)

            # set potential hook
            outhooks.append(net)

            n_filters = n_filters // 2

        # get hooks from potential hooks
        hooks = {}
        for shook, ehook in self._hook_indices.items():
            hooks[ehook] = outhooks[shook]

        return net, hooks

    def _conv_block(self, net: Tensor, n_filters: int, kernel_size: int = 3) -> Tensor:
        """
        Convolutional Block

        Parameters
        ----------

        net: Tensor
            current Tensor in the model
        n_filters: int
            current number of filters
        kernel_size: int:
            size of filter in 2d convolution


        Returns
        -------

        net: Tensor
            current Tensor of the model
        """

        # loop through number of convolutions in convolution block
        for n in range(self._n_convs):
            # apply 2D convolution
            net = Conv2D(
                n_filters,
                kernel_size,
                activation=self._activation,
                kernel_initializer="he_normal",
                padding=self._padding,
                kernel_regularizer=self._l2,
            )(net)

            # apply batch normalization
            if self._batch_norm:
                net = BatchNormalization()(net)

        return net

    def _downsample(self, net: Tensor) -> Tensor:
        """Downsampling via max pooling"""

        return MaxPooling2D(pool_size=(2, 2))(net)

    def _upsample(self, net: Tensor, n_filters: int) -> Tensor:
        """Upsamplign via nearest neightbour interpolation and additional convolution"""

        net = UpSampling2D(size=(2, 2))(net)
        net = Conv2D(
            n_filters,
            self._filter_size,
            activation=self._activation,
            padding=self._padding,
            kernel_regularizer=self._l2,
        )(net)

        return net

    def _concatenator(self, net: Tensor, item: Tensor, name="") -> Tensor:
        """"Concatenate feature maps"""

        # crop feature maps
        crop_size = int(item.shape[1] - net.shape[1]) / 2
        item_cropped = Cropping2D(int(crop_size))(item)

        if name:
            return concatenate([item_cropped, net], axis=3, name=name)
        return concatenate([item_cropped, net], axis=3)

    def _merger(self, net: Tensor, item: Tensor) -> Tensor:
        """"Combine feature maps"""

        # crop feature maps
        crop_size = int(item.shape[1] - net.shape[1]) / 2
        item_cropped = Cropping2D(int(crop_size))(item)

        # adapt number of filters via 1x1 convolutional to allow merge
        current_filters = int(net.shape[-1])
        item_cropped = Conv2D(
            current_filters, 1, activation=self._activation, padding=self._padding
        )(item_cropped)

        # Combine feature maps by adding
        if self._merge_type == "add":
            return Add()([item_cropped, net])
        # Combine feature maps by subtracting
        if self._merge_type == "subtract":
            return Subtract()([item_cropped, net])
        # Combine feature maps by multiplication
        if self._merge_type == "multiply":
            return Multiply()([item_cropped, net])

        # Raise ValueError if merge type is unsupported
        raise ValueError(f"unsupported merge type: {self._merge_type}")
