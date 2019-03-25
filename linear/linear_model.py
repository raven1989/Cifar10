import sys
import tensorflow as tf
import cifar10

class LinearSoftmax:
    
    def __init__(self, 
            batch_size, input_size, num_class, 
            learning_rate, reg_rate, 
            initializer=tf.random_normal_initializer(stddev=1e-2)):
        ### hyper
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.input_size = input_size
        self.num_class = num_class
        self.initializer = initializer
        ### input
        with tf.name_scope("inputs"):
            self.X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            self.Y = tf.placeholder(tf.int32, [None], name="input_y")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.initialize_weights()
        self.logits = self.inference()
        self.loss_val, self.data_losses = self.loss()
        self.train_op = self.train()
        with tf.name_scope("predict"):
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        with tf.name_scope("metrics"):
            self.mean_loss, self.mean_loss_update = tf.metrics.mean(self.data_losses, name="mean_loss")
            self.acc, self.acc_update = tf.metrics.accuracy(labels=self.Y, predictions=self.predictions, name="acc")
        with tf.name_scope("vitual"):
            self.visual_weights = tf.transpose(tf.reshape(tf.transpose(self.W), [self.num_class, cifar10.DEPTH, cifar10.HEIGHT, cifar10.WIDTH]), [0, 2, 3, 1])

    def initialize_weights(self):
        with tf.name_scope("fc"):
            self.W = tf.get_variable(name="W", shape=[self.input_size, self.num_class], initializer=self.initializer)
            self.b = tf.get_variable(name="b", initializer=tf.zeros(shape=[self.num_class]))

    def inference(self):
        with tf.name_scope("score"):
            logits = tf.matmul(self.X, self.W, name="logits")+self.b
        return logits

    def loss(self):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits, name="softmax_loss");
            data_loss = tf.reduce_mean(losses, name="data_loss")
            reg_loss = self.reg_rate * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'b' not in v.name])
            loss = data_loss + reg_loss
        return loss, losses

    def train(self):
        with tf.name_scope("optimizer"):
            train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=self.learning_rate, optimizer="Adam")
        return train_op

#     def eval(self, X, Y):
        # with tf.name_scope("eval"):
            # logits = tf.matmul(X, self.W) + self.b
            # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits, name="softmax_loss");
            # loss = tf.reduce_mean(losses, name="mean_loss")
            # pred = tf.argmax(logits, axis=1, name="predict")
            # acc, acc_update = tf.metrics.accuracy(labels=Y, predictions=pred, name="acc")
            # # acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(pred, tf.int32), Y), tf.float32))
        # return loss, acc, acc_update


