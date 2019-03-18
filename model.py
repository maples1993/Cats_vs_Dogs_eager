import tensorflow as tf
import tensorflow.contrib.layers as layers


def inference(inputs, is_training=True, name_scope='Simple'):
    ngf = 16

    container = tf.contrib.eager.EagerVariableStore()
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        with container.as_default():
            conv1 = layers.conv2d(inputs, ngf, 3, 1, scope='conv1')
            pool1 = layers.avg_pool2d(conv1, 2, padding='SAME', scope='pool1')

            conv2 = layers.conv2d(pool1, ngf * 2, 3, 1, scope='conv2')
            pool2 = layers.avg_pool2d(conv2, 2, padding='SAME', scope='pool2')

            conv3 = layers.conv2d(pool2, ngf * 4, 3, 1, scope='conv3')
            pool3 = layers.avg_pool2d(conv3, 2, padding='SAME', scope='pool3')

            conv4 = layers.conv2d(pool3, ngf * 8, 3, 1, scope='conv4')
            pool4 = layers.avg_pool2d(conv4, 2, padding='SAME', scope='pool4')

            flatten = layers.flatten(pool4)
            fc1 = layers.fully_connected(flatten, 128, scope='fc1')
            fc1 = layers.dropout(fc1, 0.5, is_training=is_training, scope='drop1')

            fc2 = layers.fully_connected(fc1, 128, scope='fc2')
            fc2 = layers.dropout(fc2, 0.5, is_training=is_training, scope='drop2')

            fc3 = layers.fully_connected(fc2, 128, scope='fc3')
            fc3 = layers.dropout(fc3, 0.5, is_training=is_training, scope='drop3')

            logits = layers.fully_connected(fc3, 2, activation_fn=tf.nn.sigmoid, scope='logits')

        return logits, container


if __name__ == '__main__':
    # tf.enable_eager_execution()
    # tfe = tf.contrib.eager
    sess = tf.Session()

    # container = tfe.EagerVariableStore()
    input = tf.ones([8, 128, 128, 3])
    logits = inference(input)

    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('参数数目: %d' % sess.run(paras_count))