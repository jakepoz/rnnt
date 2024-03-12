import keras

from rnnt.jasper import JasperLayer

batch_size = 2

model = keras.Sequential(
    [
        keras.Input(shape=(80, 100), batch_size=batch_size),
        JasperLayer(kernel_size=11, out_channels=64, dropout=0.2, num_sub_blocks=4),
    ]
)



model.summary()