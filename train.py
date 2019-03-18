import tensorflow as tf
import load_data
import time
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def train():
    tf.enable_eager_execution()

    BATCH_SIZE = 16
    MAX_STEP = 10000

    train_dir = './data/train'
    image_paths, image_labels = load_data.get_path_and_label(train_dir, is_random=True)
    image_label_ds = load_data.get_dataset(image_paths, image_labels, BATCH_SIZE, is_random=True)

    demo_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(strides=2, padding='same'),
        tf.keras.layers.Conv2D(16, [3, 3], padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    checkpoint_dir = './logs_2'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=demo_model)

    start = time.time()
    for step, (images, labels) in enumerate(image_label_ds.take(MAX_STEP)):
        with tf.GradientTape() as tape:
            logits = demo_model(images, training=True)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads = tape.gradient(loss, demo_model.variables)
        optimizer.apply_gradients(zip(grads, demo_model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        if step % 100 == 0:
            acc = tf.nn.in_top_k(logits, labels, 1)
            acc = tf.reduce_mean(tf.cast(acc, tf.float16))
            end = time.time()
            print('Step:%6d, loss:%.6f, accuracy:%.2f%%, time:%.2fs' % (step, loss, acc * 100, end - start))
            start = time.time()

        if step % 1000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def eval():
    tf.enable_eager_execution()

    train_dir = './data/test'
    image_paths = load_data.get_path(train_dir, is_random=True)

    demo_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(strides=2, padding='same'),
        tf.keras.layers.Conv2D(16, [3, 3], padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    checkpoint = tf.train.Checkpoint(model=demo_model)
    checkpoint.restore(tf.train.latest_checkpoint('./logs_2'))

    for image in image_paths:
        img_ori = plt.imread(image)
        img = tf.convert_to_tensor(img_ori)
        img = tf.image.resize_images(img, [128, 128])
        img = tf.reshape(img, [1, 128, 128, 3])
        img = tf.cast(img / 255.0, tf.float32)
        logits = demo_model(img, training=True)
        acc = tf.argmax(logits, axis=1)
        print(logits)
        if acc.numpy() == 0:
            label = 'cat'
        else:
            label = 'dog'
        plt.imshow(img_ori)
        plt.title(label)
        plt.show()


if __name__ == '__main__':
    # train()
    eval()