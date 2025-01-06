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

def train_model(training_data, training_labels):
    
    model = build_model(num_classes)
    
    history = compile_and_train(
        model,
        training_data,
        training_labels,
        metrics = [keras.metrics.AUC(name="auc"), "acc"],
        optimizer = keras.optimizers.Adam(),
        patience = 3,
        epochs = 10,
    )
    
    model = unfreeze(model, "block_10")
    
    fine_tune_epochs = 20
    total_epochs = history.epoch[-1] + fine_tune_epochs
    
    history_fine = compile_and_train(
        model,
        training_data,
        training_labels,
        metrics = [keras.metrics.AUC(name="auc"), "acc"],
        optimizer = keras.optimizers.Adam(learning_rate = 1e-4),
        epochs = total_epochs,
    )
    
    _, _, acc = model.evaluate(img_test, label_test)
    return np.round(acc,4)

def train_interatively(sample_splits = [0.05, 0.1, 0.25, 0.5], iter_per_split = 5):
    train_acc = []
    sample_sizes = []
    
    for fraction in sample_splits:
        print(f"Fraction Split: {fraction}")
        sample_accuracy = []
        num_samples = int(num_train_samples * fraction)
        for i in range(iter_per_split):
            print(f"Rode {i+1} de {iter_per_split}: ")
            rand_idx = np.random.randint(num_train_samples, size = num_samples)
            train_img_subset = img_train[rand_idx, :]
            train_label_subset = label_train[rand_idx, :]
            accuracy = train_model(train_img_subset, train_label_subset)
            print(f"Precisão: {accuracy}")
            sample_accuracy.append(accuracy)
        train_acc.append(sample_accuracy)
        sample_sizes.append(num_samples)
    return train_acc, sample_sizes


# Running the above function produces the following outputs
#train_acc = [
#    [0.8202, 0.7466, 0.8011, 0.8447, 0.8229],
#    [0.861, 0.8774, 0.8501, 0.8937, 0.891],
#    [0.891, 0.9237, 0.8856, 0.9101, 0.891],
#    [0.8937, 0.9373, 0.9128, 0.8719, 0.9128],
#]
#
#sample_sizes = [165, 330, 825, 1651]