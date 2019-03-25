import os

import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3

class Cifar10Dataset(object):

    def __init__(self, data_dir, subset='train', flat=False, use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.flat = flat
        self.use_distortion = use_distortion

    def get_filenames(self):
        if self.subset in ['train', 'valid', 'test']:
            return [os.path.join(self.data_dir, self.subset+".tfrecords")]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example, 
                features = {
                    'image' : tf.FixedLenFeature([], tf.string), 
                    'label' : tf.FixedLenFeature([], tf.int64),
                })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])
        if self.flat:
            image = tf.cast(image, tf.float32)/255.0
        else:
            image = tf.cast(
                        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1,2,0]), 
                        tf.float32)/255.0
        label = tf.cast(features['label'], tf.int32)

        image = self.preprocess(image)

        return image, label

    def preprocess(self, image):
        return image

    def make_batch(self, batch_size):
        filenames = self.get_filenames()
        dataset = tf.contrib.data.TFRecordDataset(filenames).repeat()

        dataset = dataset.map(self.parser, num_threads=batch_size)

        # Potentially shuffle records.
        if self.subset == 'train':
            min_queue_examples = int(
                Cifar10Dataset.num_examples_per_epoch(self.subset) * 0.4)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
        # print(iterator.output_shapes)

        return image_batch, label_batch

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 50000
        elif subset == 'valid':
            return 5000
        elif subset == 'test':
            return 5000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

