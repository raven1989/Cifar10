import tensorflow as tf

def accuracy(sess, model, x, y):
    pred = sess.run(model.predict(x))
