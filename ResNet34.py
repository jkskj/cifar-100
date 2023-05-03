import tensorflow as tf
from ResNet import ResNet34
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


# 数据预处理
def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
train_images, test_images = preprocess(train_images, test_images)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.map(preprocess).batch(200)

validation_images = train_images[45000:]
train_images = train_images[0:45000]
validation_labels = train_labels[45000:]
train_labels = train_labels[0:45000]

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

datagen.fit(train_images)


def spareCE(y_true, y_pred):
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tf.reduce_mean(sce(y_true, y_pred))


def l2_loss(my_model, weights=1e-4):
    variable_list = []
    for v in my_model.trainable_variables:
        if 'kernel' or 'bias' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights


def myLoss(y_true, y_pred):
    sce = spareCE(y_true, y_pred)
    l2 = l2_loss(my_model=model)
    loss = sce + l2
    return loss


model = ResNet34()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.004),
    loss=myLoss,
    metrics=['sparse_categorical_accuracy'])
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64), steps_per_epoch=len(train_images) / 64,
    epochs=50, verbose=2,
    validation_data=(validation_images, validation_labels)
)
model.save('saved_model/ResNet34')
my_model = tf.keras.models.load_model('saved_model/ResNet34', custom_objects={'myLoss': myLoss})

correct = 0
total = 0
for x, y in test_dataset:
    y_pred = my_model(x, training=False)
    y_pred = tf.cast(tf.argmax(y_pred, 1), dtype=tf.int32)
    y_true = tf.cast(tf.squeeze(y, -1), dtype=tf.int32)
    equality = tf.equal(y_pred, y_true)
    equality = tf.cast(equality, dtype=tf.float32)
    correct += tf.reduce_sum(equality)
    total += x.shape[0]
    print(float(correct) / total)
print('Accuracy=', float(correct) / total)
# Accuracy= 0.6106
# Accuracy= 0.6048
# Accuracy= 0.6016
fig1, ax_acc = plt.subplots()
plt.plot(history.history['sparse_categorical_accuracy'], 'r', label='acc')
plt.plot(history.history['val_sparse_categorical_accuracy'], 'b', label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(loc='lower right')
plt.savefig('./picture/ResNet34_acc.png')
plt.show()

fig2, ax_loss = plt.subplots()
plt.plot(history.history['loss'], 'r', label='loss')
plt.plot(history.history['val_loss'], 'b', label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(loc='upper right')
plt.savefig('./picture/ResNet34_loss.png')
plt.show()
