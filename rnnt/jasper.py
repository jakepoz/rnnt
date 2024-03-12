import keras

class JasperLayer(keras.Layer):
    def __init__(self, kernel_size, out_channels, dropout, num_sub_blocks):
        super(JasperLayer, self).__init__()
        self.kernel_size = kernel_size

        self.out_channels = out_channels
        self.dropout = keras.layers.Dropout(dropout)
        self.num_sub_blocks = num_sub_blocks

        self.convs = []
        self.bns = []

        for i in range(num_sub_blocks):
            self.convs.append(keras.layers.Conv1D(out_channels, kernel_size, padding='causal'))
            self.bns.append(keras.layers.BatchNormalization())

        self.residual_conv = keras.layers.Conv1D(out_channels, 1)
        self.residual_bn = keras.layers.BatchNormalization()


    def call(self, x):
        input_x = x

        residual_x = self.residual_conv(input_x)
        residual_x = self.residual_bn(residual_x)

        for index, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x)
            x = bn(x)

            if index == self.num_sub_blocks - 1:
                x = x + residual_x

            # Using GELU instead of RELU here
            x = keras.activations.gelu(x)
            x = self.dropout(x)

        return x