import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):

    def __init__(self, kernel_num, stride=1):
        super(BasicBlock, self).__init__()

        # 第一层3*3卷积网络
        self.conv1 = layers.Conv2D(kernel_num, (3, 3), strides=stride, padding='same')
        self.bin1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        # 第二层3*3卷积网络
        self.conv2 = layers.Conv2D(kernel_num, (3, 3), strides=1, padding='same')
        self.bin2 = layers.BatchNormalization()
        # 将input数据做处理，如果上面卷积结果没有降维，就不做下采样 else中的内容
        # 如果降维了就做下采样 if中的内容
        if stride != 1:
            self.downSample = Sequential()
            self.downSample.add(layers.Conv2D(kernel_num, (1, 1), strides=stride))
            # using kernel size (1,1) is not same with pooling opration
            # target is that reshape input add convolution output legitlegitimate
        else:
            self.downSample = lambda x: x

    def call(self, inputs, training=None):
        # b,h,w,c
        conv1 = self.conv1(inputs)
        bn1 = self.bin1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bin2(conv2)

        residual = self.downSample(inputs)

        output = layers.add([residual, bn2])
        output = tf.nn.relu(output)
        return output


class ResNet(keras.Model):

    def __init__(self, layers_dims, num_class=100):  # layers_dims#[2,2,2,2]
        super(ResNet, self).__init__()

        self.stem = Sequential([
            layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])
        self.layer1 = self.build_resblock(64, layers_dims[0])
        self.layer2 = self.build_resblock(128, layers_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layers_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layers_dims[3], stride=2)
        # feature size will sucessive reduce

        # output:[b,512,h(unknown),w(unknown)]=>GlobalAveragePooling2D()=>[b,512]
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_class)

    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_resblock(self, kernel_num, blocks, stride=1):
        res_blocks = Sequential()

        res_blocks.add(BasicBlock(kernel_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(kernel_num, stride=1))

        return res_blocks


def ResNet18():
    return ResNet([2, 2, 2, 2], num_class=100)


def ResNet34():
    return ResNet([3, 4, 6, 3], num_class=100)
