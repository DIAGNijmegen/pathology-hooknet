"""
This file contains a wrapper class for writing patches to multi-resolution images.
"""

import multiresolutionimageinterface as mir

import numpy as np
import time
#----------------------------------------------------------------------------------------------------

class ImageWriter(object):
    """Wrapper class for multi-resolution image writing."""

    def __init__(self, image_path, shape, spacing, dtype, coding, compression=None, interpolation=None, tile_size=512, jpeg_quality=None, empty_value=0, skip_empty=None):
        """
        Initialize the object and open the given image. Missing compression and interpolation methods and the skip empt flag are derived from the color coding: Monochrome images are
        compressed with 'lzw' and interpolated with 'nearest' method and all tiles written out. Otherwise the compression method is 'jpeg' and the interpolation is 'linear' and the skip
        empty flag is enabled.

        Args:
            image_path (str): Path of the image to write.
            shape (tuple): Shape of the image.
            spacing (float, None): Pixel spacing (micrometer).
            dtype (type): Data type. Values: np.uint8, np.uint16, np.uint32, np.float32.
            coding (str): Color coding of the pixels. Values: 'monochrome', 'rgb', or 'argb'.
            compression (str, None): Data compression method in the image file. Values: 'raw', 'jpeg', oe 'lzw'.
            interpolation (str, None): Interpolation for calculating the image pyramid in the image file. Values: 'nearest', or 'linear'.
            tile_size (int): Tile size in the image file.
            jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default 80 is used.
            empty_value (int): Value of the missing or padded tiles.
            skip_empty (bool, None): Skip writing out tiles that are filled with the empty value.

        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__writer = None         # Multi-resolution image writer object.
        self.__path = ''             # Path of the opened image.
        self.__shape = None          # Shape of the image on the first level.
        self.__spacing = None        # Pixel spacing.
        self.__dtype = None          # Pixel data type.
        self.__coding = None         # Pixel color coding method.
        self.__compression = None    # Compression method in the image file.
        self.__interpolation = None  # Interpolation method for creating the image pyramid.
        self.__tile_size = None      # Tile size for writing the image.
        self.__tile_shape = None     # Expected shape of the written tile.
        self.__jpeg_quality = None   # JPEG quality.
        self.__empty_value = 0       # Empty value for writing empty or missing parts.
        self.__empty_tile = None     # Empty tile for writing empty part.
        self.__skip_empty = None     # Skip writing out of empty tiles.

        # Open image.
        #
        self.__open(image_path=image_path)
        self.__setparameters(dtype=dtype, coding=coding, compression=compression, interpolation=interpolation, tile_size=tile_size, jpeg_quality=jpeg_quality)
        self.__setdimensions(shape=shape, spacing=spacing)
        self.__configureemptytile(empty_value=empty_value, skip_empty=skip_empty)

    def __del__(self):
        """Delete the object."""

        self.close()

    def __open(self, image_path):
        """
        Open multi-resolution image for writing.

        Args:
            image_path (str): Path of the image to load.
        """

        # Create image writer object.
        #
        self.__writer = mir.MultiResolutionImageWriter()

        # Open file for writing.
        #
        result_code = self.__writer.openFile(image_path)

        if result_code != 0:
            raise ValueError(f'error opening image') 

        # Save the path of the file.
        #
        self.__path = image_path

    def __setparameters(self, dtype, coding, compression, interpolation, tile_size, jpeg_quality):
        """
        Configure the parameters of the multi-resolution image.

        Args:
            dtype (type): Data type.
            coding (str): Color coding of the pixels.
            compression (str): Data compression method in the image file.
            interpolation (str): Interpolation for calculating the image pyramid in the image file.
            tile_size (int): Tile size in the image file.
            jpeg_quality (int): JPEG quality (1-100) when using JPEG as compression method.

        """

        # Convert data type.
        #
        if dtype == np.uint8:
            dtype_param = mir.UChar
        elif dtype == np.uint16:
            dtype_param = mir.UInt16
        elif dtype == np.uint32:
            dtype_param = mir.UInt32
        elif dtype == np.float32:
            dtype_param = mir.Float
        else:
            raise ValueError(f'Invalid data type: {dtype}') 

        # Convert color coding.
        #
        if coding == 'monochrome':
            coding_param = mir.Monochrome
            channel_count = 1
        elif coding == 'rgb':
            coding_param = mir.RGB
            channel_count = 3
        elif coding == 'argb':
            coding_param = mir.ARGB
            channel_count = 4
        else:
            raise ValueError(f'Invalid color type: {coding}') 

        # Convert compression method.
        #
        if compression is not None:
            if compression == 'raw':
                compression_param = mir.RAW
            elif compression == 'jpeg':
                compression_param = mir.JPEG
            elif compression == 'lzw':
                compression_param = mir.LZW
            else:
                raise ValueError(f'Invalid compression method: {compression}') 

            compression_save = compression
        else:
            # Derive the compression method from the color coding.
            #
            if coding == 'monochrome':
                compression_param = mir.LZW
                compression_save = 'lzw'
            else:
                compression_param = mir.JPEG
                compression_save = 'jpeg'

        # Convert interpolation method.
        #
        if interpolation is not None:
            if interpolation == 'nearest':
                interpolation_param = mir.NearestNeighbor
            elif interpolation == 'linear':
                interpolation_param = mir.Linear
            else:
                raise ValueError(f'Invalid interpolation method: {interpolation}') 

            interpolation_save = interpolation
        else:
            # Derive the interpolation method from the color coding.
            #
            if coding == 'monochrome':
                interpolation_param = mir.NearestNeighbor
                interpolation_save = 'nearest'
            else:
                interpolation_param = mir.Linear
                interpolation_save = 'linear'

        # Check tile size.
        #
        if tile_size <= 0:
            raise IOError(f'Invalid tile size: {tile_size}') 

        # Check JPEG quality setting.
        #
        if jpeg_quality is not None:
            if jpeg_quality < 1 or 100 < jpeg_quality:
                raise ValueError(f'Invalid jpeg quality size: {jpeg_quality}') 

            jpeg_quality_save = jpeg_quality
        else:
            jpeg_quality_save = 80 if compression_save == 'jpeg' else jpeg_quality

        # Save the parameters.
        #
        self.__dtype = dtype
        self.__coding = coding
        self.__compression = compression_save
        self.__interpolation = interpolation_save
        self.__tile_size = tile_size
        self.__tile_shape = (channel_count, tile_size, tile_size)
        self.__jpeg_quality = jpeg_quality_save

        # Configure parameters.
        #
        self.__writer.setDataType(dtype_param)
        self.__writer.setColorType(coding_param)
        self.__writer.setCompression(compression_param)
        self.__writer.setInterpolation(interpolation_param)
        self.__writer.setTileSize(tile_size)

        if jpeg_quality_save is not None:
            self.__writer.setJPEGQuality(jpeg_quality_save)

    def __setdimensions(self, shape, spacing):
        """
        Set the shape and the pixel spacing of the multi-resolution image.

        Args:
            shape (tuple): Shape of the image.
            spacing (float, None): Pixel spacing (micrometer).

        """

        # Check the shape.
        #
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise IOError(f'Invalid shape: {shape}') 

        # Check the spacing.
        #
        if spacing is not None and spacing <= 0.0:
            raise IOError(f'Invalid spacing: {spacing}') 

        # Save the parameters.
        #
        self.__shape = shape
        self.__spacing = spacing

        # Configure shape and pixel spacing.
        #
        self.__writer.writeImageInformation(shape[1], shape[0])

        if spacing is not None:
            pixel_size_vec = mir.vector_double()
            pixel_size_vec.push_back(spacing)
            pixel_size_vec.push_back(spacing)
            self.__writer.setSpacing(pixel_size_vec)

    def __configureemptytile(self, empty_value, skip_empty):
        """
        Configure the empty tile for writing.

        Args:
            empty_value (int): Value of the missing or padded tiles.
            skip_empty (bool, None): Skip writing out tiles that are filled with the empty value.
        """

        # Prepare empty tile.
        #
        self.__empty_value = empty_value
        self.__empty_tile = np.full(shape=self.__tile_shape, fill_value=self.__empty_value, dtype=self.__dtype)

        # Save the skip empty flag.
        #
        if skip_empty is not None:
            skip_empty_save = skip_empty
        else:
            if self.__coding == 'monochrome':
                skip_empty_save = True
            else:
                skip_empty_save = False

        self.__skip_empty = skip_empty_save

    def __rightbottompad(self, tile):
        """
        Pad the tile on the right and bottom side with the empty value.

        Args:
            tile (np.ndarray): Tile to pad.

        Returns:
            np.ndarray: Padded tile.
        """

        # Pad the tile to the target tile shape.
        #
        return np.pad(array=tile,
                      pad_width=((0, self.__tile_shape[0] - tile.shape[0]), (0, self.__tile_shape[1] - tile.shape[1]), (0, self.__tile_shape[2] - tile.shape[2])),
                      mode='constant',
                      constant_values=(self.__empty_value,))

    @property
    def path(self):
        """
        Get the path of the opened image.

        Returns:
            str: Path of the opened image.
        """

        return self.__path

    @property
    def shape(self):
        """
        Get the shape of the image at the lowest level.

        Returns:
            tuple: Image shape.
        """

        return self.__shape

    @property
    def spacing(self):
        """
        Get the pixel spacing of the image at the lowest level.

        Returns:
            float: Pixel spacing.
        """

        return self.__spacing

    @property
    def dtype(self):
        """
        Get the pixel data type.

        Returns:
            type: Pixel type.
        """

        return self.__dtype

    @property
    def coding(self):
        """
        Get the color coding. Possible values are: 'monochrome', 'rgb', 'argb', and 'indexed'.

        Returns:
            str: Color coding identifier.
        """

        return self.__coding

    @property
    def compression(self):
        """
        Get the image compression method. Possible values are: 'raw', 'jpeg', and 'lzw'.

        Returns:
            str: Compression method.
        """

        return self.__compression

    @property
    def interpolation(self):
        """
        Get the image interpolation method. Possible values are: 'nearest', and 'linear'.

        Returns:
            str: Interpolation method.
        """

        return self.__interpolation

    @property
    def tilesize(self):
        """
        Get the tile size.

        Returns:
            int: Tile size.
        """

        return self.__tile_size

    @property
    def tileshape(self):
        """
        Get the tile shape.

        Returns:
            int: Tile size.
        """

        return self.__tile_shape

    @property
    def quality(self):
        """
        Get the JPEG quality setting. It is only used when JPEG compression is set.

        Returns:
            int, None: JPEG quality.
        """

        return self.__jpeg_quality

    @property
    def emptyvalue(self):
        """
        Get the empty value.

        Returns:
            int: The value of empty or missing areas.
        """

        return self.__empty_value

    @property
    def skipempty(self):
        """
        Get the flag of skipping empty tiles.

        Returns:
            bool: Skip empty tiles flag.
        """

        return self.__skip_empty

    def fill(self, content):
        """
        Write out an array as image.

        Args:
            content (np.ndarray): Image content to write.
        """

        # Check content shape.
        #
        if content.shape[-2:] != self.__shape:
            raise IOError(f'invalid shape {content.shape[-2:], self.__shape}')

        if not (content.shape[0] == 1 and self.__coding == 'monochrome' or
                content.shape[0] == 3 and self.__coding == 'rgb' or
                content.shape[0] == 4 and self.__coding == 'argb'):

            raise IOError(f'invalid shape {content.shape, self.__coding}')

        # Write out the image content tile by tile.
        #
        for row in range(0, content.shape[1], self.__tile_size):
            for col in range(0, content.shape[2], self.__tile_size):
                self.write(tile=content[:, row:row + self.__tile_size, col: col + self.__tile_size], row=row, col=col)

    def write(self, tile, row, col):
        """
        Write the next tile to the image. Tiles must be written in order by filling up the rows continuously. Empty tiles are replaced with empty value. Tiles smaller than the required size
        are padded. Tiles are expected in channels first order.

        The target position of the tile can be added with the 'row' and 'col' pixel addresses. The 'row' and 'col' values have to be either both set or both None. The given coordinates means
        the upper left corner of the tile to write. It is not recommended to switch between automatic and explicit addressing.

        Args:
            tile (np.ndarray, None): Tile to write. None or empty tiles are replaced with an empty value.
            row (int): Row index of upper left pixel.
            col (int): Column index of upper left pixel.

        """

        # Check if the image is still open.
        #
        if self.__writer is None:
            raise IOError(f'image: {self.__path} already closed')

        # Check the coordinates.
        #
        if row % self.__tile_size != 0 or col % self.__tile_size != 0:
            raise IOError(f'invalid tile address {(row, col)}')

        # Check data type.
        #
        if tile is not None and tile.dtype != self.__dtype:
            raise IOError(f'invalid type {(self.__dtype, tile.dtype)}')

        # Check the input tile.
        #
        if tile is None or tile.size == 0:
            # Write out an empty (tile filled with the empty value) tile if the passed tile is None or empty sized.
            #
            if not self.__skip_empty:
                self.__writer.writeBaseImagePartToLocation(self.__empty_tile.flatten(), col, row)
        elif tile.ndim == len(self.__tile_shape):
            # Check if the tile is filled with the empty value.
            #
            if not self.__skip_empty or not np.array_equal(tile, self.__empty_tile):
                # Write out the given tile with padding if necessary.
                #
                if tile.shape != self.__tile_shape:
                    raise IOError(f'invalid tile address {(row, col)}')
                else:
                    self.__writer.writeBaseImagePartToLocation(tile, col, row)
        else:
            # The dimension count does not match.
            #
            raise ValueError(f'The dimension count does not match: {tile.shape}')

    """Close the image object. No further writing is possible after calling this function."""
    def close(self):
        if self.__writer is not None:
            self.__writer.finishImage()
            time.sleep(30)
            self.__writer = None
