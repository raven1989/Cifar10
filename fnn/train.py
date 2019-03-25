# -*- coding: utf-8 -*-
import sys
sys.path.append("../util")
import os
import logging
import numpy as np
import tensorflow as tf
from cifar10 import Cifar10Dataset
import cifar10
from fnn import FNN

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_path", "../data", "Data dir")
tf.app.flags.DEFINE_string("train_data_files", "train.tfrecords", "Train data files")
tf.app.flags.DEFINE_string("valid_data_files", "valid.tfrecords", "Validation data files")
tf.app.flags.DEFINE_string("test_data_files", "test.tfrecords", "Test data files")
tf.app.flags.DEFINE_boolean("is_training", "True", "Is training or testing")

tf.app.flags.DEFINE_integer("epoch", 700, "Epoch num")
tf.app.flags.DEFINE_integer("validate_every", 1, "Epoch num")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size")
tf.app.flags.DEFINE_integer("num_class", 10, "Number of classes")
tf.app.flags.DEFINE_integer("num_hidden_layer", 2, "Number of hidden layers")
tf.app.flags.DEFINE_integer("num_hidden_unit", 100, "Number of hidden layers")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
tf.app.flags.DEFINE_float("reg_rate", 1e-2, "Learning rate")
tf.app.flags.DEFINE_string("ckpt_dir", "./checkpoint", "Checkpoint location for the model")
tf.app.flags.DEFINE_string("log_dir", "./logs", "Logs location for the model")

tf.app.flags.DEFINE_string("visible_gpus", "0", "gpus to use, default to 0.")

def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_gpus

    logging.info("Train files:{}".format(FLAGS.train_data_files))
    logging.info("Valid files:{}".format(FLAGS.valid_data_files))
    logging.info("Test files:{}".format(FLAGS.test_data_files))

    if FLAGS.is_training:
        train_dataset = Cifar10Dataset(FLAGS.data_path, subset="train", flat=True, use_distortion=False)
        valid_dataset = Cifar10Dataset(FLAGS.data_path, subset="valid", flat=True, use_distortion=False)

        logging.info("Batch size:{}".format(FLAGS.batch_size))

        train_x, train_y = train_dataset.make_batch(FLAGS.batch_size)
        valid_x, valid_y = valid_dataset.make_batch(FLAGS.batch_size)


        input_size = train_x.shape[1]

    else:
        test_dataset = Cifar10Dataset(FLAGS.data_path, subset="test", flat=True, use_distortion=False)

        test_x, test_y = test_dataset.make_batch(FLAGS.batch_size)

        input_size = test_x.shape[1]

    model = FNN(
                FLAGS.batch_size, input_size, FLAGS.num_class, 
                FLAGS.num_hidden_layer, FLAGS.num_hidden_unit, 
                FLAGS.learning_rate, FLAGS.reg_rate)

    if FLAGS.is_training:
        summary_metrics = []
        summary_metrics.append(tf.summary.scalar("data_loss", model.mean_loss))
        summary_metrics.append(tf.summary.scalar("accuracy", model.acc))

        summary_op = tf.summary.merge(inputs=summary_metrics)

        summary_visual_weights = tf.summary.image("visual_weights", model.visual_weights, max_outputs=FLAGS.num_hidden_unit)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        
        saver = tf.train.Saver()

        train_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "train"), graph=sess.graph)
        valid_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "validate"), graph=sess.graph)

        if os.path.exists(os.path.join(FLAGS.ckpt_dir, "checkpoint")):
            logging.info("Restore Variables from checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            logging.info('Initializing Variables')
            sess.run(tf.global_variables_initializer())

        batch_size = FLAGS.batch_size

        if FLAGS.is_training:
            num_per_epoch = Cifar10Dataset.num_examples_per_epoch(subset='train')
            cur_epoch = sess.run(model.epoch_step)
            for epoch in range(cur_epoch, FLAGS.epoch):
                logging.info("Epoch:{} starts".format(epoch))
                sess.run(tf.local_variables_initializer())
                cnt = 0
                train_loss, train_acc = 0, 0
                while cnt < num_per_epoch:
                    ### really fetch mini batch into mem
                    batch_x, batch_y = sess.run([train_x, train_y])

                    _, loss, train_loss, train_acc, summary_str = sess.run([model.train_op, model.loss_val, model.mean_loss_update, model.acc_update, summary_op], feed_dict={model.X:batch_x, model.Y:batch_y})
                    if (cnt/batch_size)%50 == 0:
                        logging.info("Epoch:{}\tBatch:{}\tTrain Loss:{:.6f}\t Train Data Loss:{:.6f}".format(epoch, cnt, loss, train_loss))

                    cnt += batch_x.shape[0]

                # summary_str = sess.run(summary_op)
                train_summary_writer.add_summary(summary_str, epoch)

                sess.run(model.epoch_increment)

                if epoch%FLAGS.validate_every == 0 or epoch+1 == FLAGS.epoch:
                    ### clear metrics
                    sess.run(tf.local_variables_initializer())

                    cnt = 0
                    valid_loss, valid_acc = 0, 0

                    valid_num_per_epoch = Cifar10Dataset.num_examples_per_epoch(subset='valid')
                    while cnt < valid_num_per_epoch:
                        ### really fetch mini batch into mem
                        batch_x, batch_y = sess.run([valid_x, valid_y])

                        valid_loss, valid_acc, summary_str, summary_img = sess.run([model.mean_loss_update, model.acc_update, summary_op, summary_visual_weights], feed_dict={model.X:batch_x, model.Y:batch_y})
                        cnt += batch_x.shape[0]

                    logging.info("Epoch:{}\tTrain Data Loss:{:.6f}\tValidate Loss:{:.6f}\tTrain Acc:{:.6f}\tValidate Acc:{:.6f}".format(epoch, train_loss, valid_loss, train_acc, valid_acc))

                    # summary_str = sess.run(summary_op)
                    valid_summary_writer.add_summary(summary_str, epoch)
                    if epoch+1 == FLAGS.epoch:
                        valid_summary_writer.add_summary(summary_img, epoch)

                    save_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
                    saver.save(sess, save_path, global_step=epoch)
        else: 
            sess.run(tf.local_variables_initializer())

            num_per_epoch = Cifar10Dataset.num_examples_per_epoch(subset='test')
            cnt = 0
            acc = 0
            while cnt < num_per_epoch:
                ### really fetch mini batch into mem
                batch_x, batch_y = sess.run([test_x, test_y])
                labels, predictions, acc = sess.run([model.Y, model.predictions, model.acc_update], feed_dict={model.X:batch_x, model.Y:batch_y})
                # logging.info(labels)
                # logging.info(predictions)
                cnt += batch_x.shape[0]
            logging.info("Test Acc:{:.6f}".format(acc))


if __name__ == '__main__':
    tf.app.run()


