"""Estimate required training-set size vs accuracy (tf_flowers + MobileNetV2).

Import-safe: importing this module defines helpers only. The full pipeline
(dataset download + training) runs under ``main()``::

    python estimating.py
    python estimating.py --fraction-sweep   # also train on data subsets
"""

import argparse
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


def dataset_to_array(dataset, image_size, num_classes):
    """Materialize a tfds split into (images, one-hot labels) numpy arrays.

    Resize + one-hot happen inside a batched/prefetched ``tf.data`` pipeline
    (bounded memory, vectorized kernels) instead of a per-element Python loop
    calling ``.numpy()`` on every sample — which built one TF graph per image
    and held intermediate copies.
    """
    ds = (
        dataset.map(
            lambda img, lab: (
                tf.image.resize(img, image_size),
                tf.one_hot(lab, num_classes, dtype=tf.float32),
            ),
            num_parallel_calls=AUTO,
        )
        .batch(256)
        .prefetch(AUTO)
    )
    images_parts, labels_parts = [], []
    for batch_images, batch_labels in ds.as_numpy_iterator():
        images_parts.append(np.asarray(batch_images, dtype=np.float32))
        labels_parts.append(np.asarray(batch_labels, dtype=np.float32))
    return np.concatenate(images_parts), np.concatenate(labels_parts)


def build_augmentation():
    """On-model augmentation: runs per-batch, training-mode only.

    Applied as the first layers of the classifier (instead of augmenting the
    numpy array once offline), so every epoch sees fresh variants and the
    validation/test path is never augmented.
    """
    return keras.Sequential(
        [
            layers.RandomFlip(mode="horizontal"),
            layers.RandomRotation(factor=0.1),
            layers.RandomZoom(height_factor=(-0.1, 0.1)),
            layers.RandomContrast(factor=0.1),
        ],
        name="augmentation",
    )


def build_model(num_classes, img_size=image_size[0], top_dropout=0.3):
    """Cria um classificador baseado em MobileNetV2 pré-treinado.

    Argumentos:
        num_classes: Int, número de classes a serem usadas na camada softmax.
        img_size: Int, tamanho quadrado das imagens de entrada (o padrão é 224).
        top_dropout: Int, valor para camada de eliminação (o padrão é 0,3).
    """

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = build_augmentation()(inputs)
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(x)
    model = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_tensor=x
    )

    model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avh_pool")(model.output)
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
    metrics=[keras.metrics.AUC(name="auc"), "acc"],
    optimizer=keras.optimizers.Adam(),
    patience=5,
    epochs=5,
    initial_epoch=0,
):
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        min_delta=0,
        patience=patience,
        verbose=1,
        restore_best_weights=True,
    )

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    history = model.fit(
        x=training_data,
        y=training_labels,
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_split=0.1,
        callbacks=[stopper],
    )
    return history


def unfreeze(model, block_name, verbose=0):
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


def train_model(training_data, training_labels, test_data, test_labels,
                epochs=10, fine_tune_epochs=20):

    model = build_model(training_labels.shape[1])

    history = compile_and_train(
        model,
        training_data,
        training_labels,
        metrics=[keras.metrics.AUC(name="auc"), "acc"],
        optimizer=keras.optimizers.Adam(),
        patience=3,
        epochs=epochs,
    )

    model = unfreeze(model, "block_10")

    # Continua de onde a primeira fase parou: sem initial_epoch o fit
    # recomeçaria na época 0 (curvas/history inconsistentes).
    initial_epoch = history.epoch[-1] + 1
    total_epochs = initial_epoch + fine_tune_epochs

    history_fine = compile_and_train(
        model,
        training_data,
        training_labels,
        metrics=[keras.metrics.AUC(name="auc"), "acc"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        epochs=total_epochs,
        initial_epoch=initial_epoch,
    )

    _, _, acc = model.evaluate(test_data, test_labels)
    return np.round(acc, 4)


def stratified_subset_indices(class_ids, num_samples, rng):
    """Índices sem reposição, proporcionais às classes (sem reposição).

    Substitui ``randint(n, size=k)`` — que sorteava com reposição (amostras
    repetidas + classes minoritárias podiam sumir) — por cotas por classe
    via ``choice(..., replace=False)``.
    """
    class_ids = np.asarray(class_ids)
    classes, counts = np.unique(class_ids, return_counts=True)
    total = len(class_ids)
    num_samples = int(min(num_samples, total))

    # Cotas proporcionais (pelo menos 1 por classe com disponibilidade).
    quotas = np.maximum(1, np.floor(num_samples * counts / total).astype(int))
    quotas = np.minimum(quotas, counts)
    # Distribui o resto para as classes com maior folga.
    remainder = num_samples - int(quotas.sum())
    if remainder > 0:
        slack = counts - quotas
        for cls_idx in np.argsort(-slack, kind="stable"):
            if remainder <= 0:
                break
            take = int(min(remainder, slack[cls_idx]))
            quotas[cls_idx] += take
            remainder -= take

    parts = [
        rng.choice(np.flatnonzero(class_ids == cls), size=int(q), replace=False)
        for cls, q in zip(classes, quotas)
        if q > 0
    ]
    indices = np.concatenate(parts)
    rng.shuffle(indices)
    return indices


def train_interactively(img_train, label_train, test_data, test_labels,
                        sample_splits=[0.05, 0.1, 0.25, 0.5], iter_per_split=5,
                        epochs=10, fine_tune_epochs=20):
    train_acc = []
    sample_sizes = []
    num_train_samples = len(img_train)
    class_ids = np.argmax(label_train, axis=1)
    rng = np.random.default_rng(seed)

    for fraction in sample_splits:
        print(f"Fraction Split: {fraction}")
        sample_accuracy = []
        num_samples = int(num_train_samples * fraction)
        for i in range(iter_per_split):
            print(f"Rode {i+1} de {iter_per_split}: ")
            rand_idx = stratified_subset_indices(class_ids, num_samples, rng)
            train_img_subset = img_train[rand_idx, :]
            train_label_subset = label_train[rand_idx, :]
            accuracy = train_model(train_img_subset, train_label_subset,
                                   test_data, test_labels,
                                   epochs=epochs, fine_tune_epochs=fine_tune_epochs)
            print(f"Precisão: {accuracy}")
            sample_accuracy.append(accuracy)
        train_acc.append(sample_accuracy)
        sample_sizes.append(num_samples)
    return train_acc, sample_sizes


# Running the above function produces the following outputs
train_acc = [
    [0.8202, 0.7466, 0.8011, 0.8447, 0.8229],
    [0.861, 0.8774, 0.8501, 0.8937, 0.891],
    [0.891, 0.9237, 0.8856, 0.9101, 0.891],
    [0.8937, 0.9373, 0.9128, 0.8719, 0.9128],
]

sample_sizes = [165, 330, 825, 1651]


def fit_and_predict(train_acc, sample_sizes, pred_sample_size):
    # README promete ajuste em log-space via np.polyfit: log(acc) = log(a) + b*log(n).
    # O caminho anterior fazia `a * x**b` com x=list (TypeError) e init a=b=0.0
    # (gradiente zero em a -> fit morto). Usa float32 + init sensato e cai para
    # polyfit em log-space, que é determinístico e não precisa de 5000 épocas.
    x = np.asarray(sample_sizes, dtype=np.float32)
    mean_acc = np.asarray([np.mean(i) for i in train_acc], dtype=np.float32)
    error = [float(np.std(i)) for i in train_acc]

    mse = keras.losses.MeanSquaredError()

    def exp_func(x_, a, b):
        return a * np.power(np.asarray(x_, dtype=np.float32), b)

    # Log-space linear fit: log(acc) ~ log(a) + b*log(n) (máscara acc>0, n>0).
    mask = (x > 0) & (mean_acc > 0)
    if int(np.count_nonzero(mask)) >= 2:
        slope, intercept = np.polyfit(
            np.log(x[mask].astype(np.float64)), np.log(mean_acc[mask].astype(np.float64)), 1
        )
        a_init, b_init = float(np.exp(intercept)), float(slope)
    else:
        a_init, b_init = 0.8, 0.1

    a = tf.Variable(a_init, dtype=tf.float32)
    b = tf.Variable(b_init, dtype=tf.float32)
    learning_rate = 0.01
    training_epochs = 5000

    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(mean_acc, dtype=tf.float32)
    for epoch in range(training_epochs):
        with tf.GradientTape() as tape:
            y_pred = a * tf.pow(x_tf, b)
            cost_function = mse(y_tf, y_pred)
        gradients = tape.gradient(cost_function, [a, b])
        if gradients[0] is not None:
            a.assign_sub(gradients[0] * learning_rate)
        if gradients[1] is not None:
            b.assign_sub(gradients[1] * learning_rate)
    print(f"Curve fit weights: a = {a.numpy()} e b = {b.numpy()}. ")

    max_acc = float(exp_func(np.asarray([pred_sample_size], dtype=np.float32), a.numpy(), b.numpy())[0])

    print(f"O modelo previu {pred_sample_size} samples com {max_acc} de precisão")
    x_cont = np.linspace(float(x[0]), pred_sample_size, 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, mean_acc, yerr=error, fmt="o", label="ACC médio & std dev.")
    ax.plot(x_cont, exp_func(x_cont, a.numpy(), b.numpy()), "r-", label="Curva exponencial.")
    ax.set_ylabel("Precisão da classificação do modelo.", fontsize=12)
    ax.set_xlabel("Tamanho do sample de treinamento.", fontsize=12)
    ax.set_xticks(np.append(x, pred_sample_size))
    ax.set_yticks(np.append(mean_acc, max_acc))
    ax.set_xticklabels(list(np.append(x, pred_sample_size)), rotation=90, fontsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_title("Curva de aprendizado: Precisão do modelo vs tamanho das samples.", fontsize=14)
    ax.legend(loc=(0.75, 0.75), fontsize=10)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()

    mae = keras.losses.MeanAbsoluteError()
    print(f"O MAE para o fit da curva é {mae(y_tf, a * tf.pow(x_tf, b)).numpy()}.")


def load_data():
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

    img_train, label_train = dataset_to_array(train_data, image_size, num_classes)
    img_test, label_test = dataset_to_array(test_data, image_size, num_classes)

    num_train_samples = len(img_train)
    print(f"Number of training sample: {num_train_samples}")
    return img_train, label_train, img_test, label_test, num_train_samples


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Estima tamanho de amostra necessário (tf_flowers + MobileNetV2)."
    )
    parser.add_argument(
        "--fraction-sweep", action="store_true",
        help="Também treina nos subconjuntos (lento); sem isso usa os valores de referência.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Épocas da fase frozen.")
    parser.add_argument("--fine-tune-epochs", type=int, default=20, help="Épocas extras do fine-tuning.")
    args = parser.parse_args(argv)

    img_train, label_train, img_test, label_test, num_train_samples = load_data()

    if args.fraction_sweep:
        acc, sizes = train_interactively(
            img_train, label_train, img_test, label_test,
            epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs,
        )
    else:
        acc, sizes = train_acc, sample_sizes

    fit_and_predict(acc, sizes, pred_sample_size=num_train_samples)

    accuracy = train_model(img_train, label_train, img_test, label_test,
                           epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs)
    print(f"O modelo atinge uma precisão de {accuracy} com {num_train_samples} imagens.")


if __name__ == "__main__":
    main()
