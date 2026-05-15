import json
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, callbacks, optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model


tf.keras.utils.set_random_seed(42)

CLASS_NAMES = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'cifar10_cnn.keras')
HISTORY_PATH = os.path.join(HERE, 'cifar10_cnn_history.json')

(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

val_size = 5000
X_val = X_train_full[-val_size:]
y_val = y_train_full[-val_size:]
X_train = X_train_full[:-val_size]
y_train = y_train_full[:-val_size]


def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.MaxPooling2D(2, name=f'{name}_pool')(x)
    x = layers.Dropout(0.25, name=f'{name}_drop')(x)
    return x


def build_model():
    inp = Input(shape=(32, 32, 3), name='image')
    aug = layers.RandomFlip('horizontal', name='aug_flip')(inp)
    aug = layers.RandomTranslation(0.1, 0.1, name='aug_shift')(aug)

    h = conv_block(aug, 32, 'block1')
    h = conv_block(h, 64, 'block2')
    h = conv_block(h, 128, 'block3')

    h = layers.Flatten(name='flatten')(h)
    h = layers.Dense(128, activation='relu', name='dense_1')(h)
    h = layers.BatchNormalization(name='dense_1_bn')(h)
    h = layers.Dropout(0.5, name='dense_1_drop')(h)
    out = layers.Dense(10, activation='softmax', name='proba')(h)

    m = Model(inputs=inp, outputs=out, name='cifar10_cnn')
    m.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m


if os.path.exists(MODEL_PATH) and os.path.exists(HISTORY_PATH):
    print(f"Loading cached model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    with open(HISTORY_PATH, 'r') as f:
        history_dict = json.load(f)
else:
    model = build_model()
    model.summary()

    early = callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                    restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                            patience=4, min_lr=1e-5)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=30, batch_size=128,
                        callbacks=[early, reduce_lr], verbose=2)
    history_dict = history.history

    model.save(MODEL_PATH)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history_dict, f)
    print(f"Model saved to {MODEL_PATH}")
    print(f"History saved to {HISTORY_PATH}")

model.summary()
try:
    plot_model(model, to_file=ARCH_PATH, show_shapes=True,
               show_layer_names=True, dpi=120)
    arch_plot_ok = True
except Exception as e:
    print(f"plot_model failed (graphviz/pydot missing?): {e}")
    arch_plot_ok = False

epochs_run = len(history_dict['loss'])
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
test_err = 1.0 - test_acc

print(f"Epochs run    : {epochs_run}")
print(f"Test loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_acc:.4f}")
print(f"Test error    : {test_err:.4f}")

y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)

print("\nPer-class accuracy:")
for i, name in enumerate(CLASS_NAMES):
    row = cm[i]
    acc_i = row[i] / row.sum() if row.sum() else 0.0
    print(f"  {name:<6s}: {acc_i:.3f}  ({row[i]}/{row.sum()})")

plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

axes[0].plot(history_dict['loss'], label='train loss')
axes[0].plot(history_dict['val_loss'], label='val loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Cross-entropy')
axes[0].set_title(f'Loss vs. Epochs, Final test loss: {round(test_loss, 4)}')
axes[0].legend()

axes[1].plot(history_dict['accuracy'], label='train acc')
axes[1].plot(history_dict['val_accuracy'], label='val acc')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].set_title(f'Accuracy vs. Epochs, Test acc: {round(test_acc, 3)}')
axes[1].legend()

im = axes[2].imshow(cm, cmap='Blues')
axes[2].set_xticks(range(10))
axes[2].set_yticks(range(10))
axes[2].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
axes[2].set_yticklabels(CLASS_NAMES)
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')
axes[2].set_title(f'Confusion matrix (test set)\n'
                  f'Epochs={epochs_run}, test acc={round(test_acc, 3)}, '
                  f'test err={round(test_err, 3)}')
for i in range(10):
    for j in range(10):
        axes[2].text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black',
                     fontsize=8)
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle('CIFAR-10 CNN — Keras Functional API '
             '(3x [Conv-BN-Conv-BN-Pool-Drop] -> Dense128 -> Softmax10)',
             fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.95))

plt.show()
