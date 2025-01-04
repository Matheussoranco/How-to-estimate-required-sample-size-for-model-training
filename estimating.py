import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import tensorflow_datasets as tfds

seed = 42
keras.utils.set_random_seed(seed)
AUTO = tf.data.AUTOTUNE

dataset_name = "tf_flowers"
batch_size = 64
image_size = (224, 224)

(train_data, test_data), ds_info = tfds.load(
    dataset_name,
    split=["train[:90%]", "train[90%:]"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

num_classes = ds_info.features["label"].num_classes
class_names = ds_info.features["label"].names

print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")


def dataset_to_array(dataset, image_size, num_classes):
    images, labels = [], []
    for img, lab in dataset.as_numpy_iterator():
        images.append(tf.image.resize(img, image_size).numpy())
        labels.append(tf.one_hot(lab, num_classes))
    return np.array(images), np.array(labels)


img_train, label_train = dataset_to_array(train_data, image_size, num_classes)
img_test, label_test = dataset_to_array(test_data, image_size, num_classes)

num_train_samples = len(img_train)
print(f"Number of training sample: {num_train_samples}")

image_augmentation = keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.1),
        layers.RandomZoom(height_factor=(-0.1, -0)),
        layers.RandomContrast(factor=0.1),
    ],
)

img_train = image_augmentation(img_train).numpy()

def build_model(num_classes, img_size=image_size[0], top_dropout=0.3):
    """Cria um classificador baseado em MobileNetV2 pré-treinado.

    Argumentos:
        num_classes: Int, número de classes a serem usadas na camada softmax.
        img_size: Int, tamanho quadrado das imagens de entrada (o padrão é 224).
        top_dropout: Int, valor para camada de eliminação (o padrão é 0,3).
    """
        
    inputs = layers.Imput(shape=(img_size, img_size, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)
    model = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_tensor=x
    )
    
    model.trainable = False
    
    x= layers.GlobalAveragePooling2D(name="avh_pool")(model.output)
    x = layers.Dropout(top_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    
    print("Weights treináveis:", len(model.trainable_weights))
    print("Weights não treináveis:", len(model.non_trainable_weights))
    return model


def compile_and_train(
    model,
    training_data,
    training_labels,
    metrics = [keras.metrics.AUC(name="auc"), "acc"],
    optimizer = keras.optimizers.Adam(),
    patience = 5,
    epochs = 5,
):
    stopper = keras.callbacks.EarlyStopping(
        monitor = "val_auc",
        mode = "max",
        min_delta = 0,
        patience = patience,
        verbose = 1,
        restore_best_weights = True,
    )
    
    model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = metrics)
    
    history = model.fit(
        x = training_data,
        y = training_labels,
        batch_size = batch-size,
        epochs = epochs,
        validation_split = 0.1,
        callbacks = [stopper],
    )
    return history

def unfreeze(model, block_name, verbose = 0):
    """Unfreeze as camadas do modelo do keras"""
    
    set_trainable = False
    
    for layer in model.layers:
        if block_name in layer.name:
            set_trainable = True
        if set_trainable and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
            if verbose == 1:
                print(layer.name, "Treinável")
        else:
            if verbose == 1:
                print(layer.name, "NÃO treinável")
    print("Weights treináveis:", len(model.trainable_weights))
    print("Weights não treináveis:", len(model.non_trainable_weights))
    return model