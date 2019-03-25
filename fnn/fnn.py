import tensorflow as tf
import cifar10

class FNN:

    def __init__(self, 
            batch_size, input_size, num_class, 
            num_hidden_layer, num_hidden_unit, 
            learning_rate, reg_rate, 
            initializer=tf.random_normal_initializer(stddev=1e-2)):
        ### hyper
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.input_size = input_size
        self.num_class = num_class
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_unit = num_hidden_unit
        self.initializer = initializer
        ### input
        with tf.name_scope("input"):
            self.X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            self.Y = tf.placeholder(tf.int32, [None], name="input_y")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))

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
            self.visual_weights = tf.transpose(tf.reshape(tf.transpose(self.W[0]), [self.num_hidden_unit, cifar10.DEPTH, cifar10.HEIGHT, cifar10.WIDTH]), [0, 2, 3, 1])

    def initialize_weights(self):
        self.W = []
        self.b = []
        m, n = self.input_size, self.num_hidden_unit
        for layer in range(self.num_hidden_layer):
            with tf.name_scope("hidden_%d" % layer):
                self.W.append(tf.get_variable(name="W_%d" % layer, shape=[m, n], initializer=self.initializer))
                self.b.append(tf.get_variable(name="b_%d" % layer, shape=[n], initializer=self.initializer))
            m = n
        with tf.name_scope("output"):
            self.W.append(tf.get_variable(name="W_output", shape=[m, self.num_class], initializer=self.initializer))
            self.b.append(tf.get_variable(name="b_output", shape=[self.num_class], initializer=self.initializer))

    def inference(self):
        self.h = []
        x = self.X
        for layer in range(self.num_hidden_layer):
            with tf.name_scope("hidden_%d" % layer):
                z = tf.matmul(x, self.W[layer])+self.b[layer]
                h = tf.nn.relu(z, name="h_%d" % layer)
                self.h.append(h)
            x = h
        with tf.name_scope("output"):
            logits = tf.matmul(x, self.W[-1]) + self.b[-1]
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


