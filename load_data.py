import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt


def get_path(root_dir, is_random=True):
    root_dir = pathlib.Path(root_dir)
    image_paths = list(map(str, root_dir.glob('*.jpg')))

    if is_random:
        random.shuffle(image_paths)
    return image_paths


# 获取图像路径及标签
def get_path_and_label(root_dir, is_random=True):
    image_paths = get_path(root_dir, is_random)

    image_labels = []
    cat_count = 0
    dog_count = 0
    for path in image_paths:
        if path.split('\\')[-1].startswith('cat'):
            image_labels.append(0)
            cat_count += 1
        else:
            image_labels.append(1)
            dog_count += 1

    print('%s中包含\t%d张图片\ndog\t%d张\ncat\t%d张.' % (root_dir, len(image_paths), dog_count, cat_count))

    return image_paths, image_labels


# 图像预处理
def preprocess_image(raw_image):
    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = tf.image.resize_images(image, [128, 128])
    image /= 255.0
    return image


# 加载图像
def load_and_preprocess_image(img_path):
    raw_image = tf.read_file(img_path)
    return preprocess_image(raw_image)


# 加载图像和标签
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


# 构建Dataset
def get_dataset(image_paths, image_labels, batch_size, is_random=True):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, tf.cast(image_labels, tf.int64)))
    ds = ds.map(load_and_preprocess_from_path_label, num_parallel_calls=2)

    if is_random:
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=200))

    ds = ds.batch(batch_size)
    return ds


if __name__ == '__main__':
    tf.enable_eager_execution()

    BATCH_SIZE = 9
    MAX_STEP = 10

    root_dir = './data/train'
    image_paths, image_labels = get_path_and_label(root_dir)
    image_label_ds = get_dataset(image_paths, image_labels, BATCH_SIZE)
    print(image_label_ds)

    for images, labels in image_label_ds.take(MAX_STEP):
        for i in range(BATCH_SIZE):
            plt.subplot(3, 3, i+1)
            plt.imshow(images.numpy()[i])
            plt.title(labels.numpy()[i])
        plt.show()
        print(images.shape)
