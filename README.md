# how_to_optimize_convolution_in_CPU

nchw conv:
        INPUT = im2col(input)
        OUTPUT = W x INPUT

nhwc conv:
        INPUT = im2row(input)
        OUTPUT = INPUT X WEIGHT_T