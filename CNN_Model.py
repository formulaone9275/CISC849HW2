from __future__ import print_function,division
import tensorflow as tf
from padding_image import padding_image,iter_dataset
import matplotlib.pyplot as plt
import os
import pickle

class CNNModel(object):

    def __init__(self,file_path,batch_size):
        self.file_path=file_path
        self.batch_size=batch_size
        self.train_epoch=15
        self.sess=tf.Session()
        self.depth_image=False
        self.down_sample=False
        self.pixel_interval=1

    def build(self):

        self.x = tf.placeholder(tf.float32, shape=[None, 350, 430,3])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 4])
        self.regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4)
        self.drop_prob = tf.placeholder(tf.float32)
        self.drop_prob_dense = tf.placeholder(tf.float32)
        self.IsTraining = tf.placeholder(tf.bool)

        # Convolutional Layer #1

        conv1 = tf.layers.conv2d(
            inputs=self.x,
            filters=64,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu,
            kernel_regularizer =self.regularizer)


        print(conv1.get_shape())

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=2,padding='same')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        # Convolutional Layer #12
        conv2 = tf.layers.conv2d(
            inputs=norm1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
            kernel_regularizer =self.regularizer)
        print(conv2.get_shape())

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[3,3], strides=2,padding='same')

        print(pool2.get_shape())

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 88*108*64])
        #
        print(pool2_flat.get_shape())


        dropout = tf.layers.dropout(
            inputs=pool2_flat, rate=self.drop_prob,training=self.IsTraining)

        dense1 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu,
                                 kernel_regularizer =self.regularizer)


        dropout1 = tf.layers.dropout(
            inputs=dense1, rate=self.drop_prob_dense,training=self.IsTraining)

        dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu,
                                 kernel_regularizer =self.regularizer)


        dropout2 = tf.layers.dropout(
            inputs=dense2, rate=self.drop_prob_dense,training=self.IsTraining)
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout2, units=4)

        self.y = tf.nn.softmax(logits)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))

        #train_step = tf.train.AdamOptimizer(7e-4).minimize(cross_entropy)
        self.y_p = tf.argmax(self.y, 1)
        self.y_t = tf.argmax(self.y_, 1)
        #calculate the precision, recall and F score
        acc, acc_op = tf.metrics.accuracy(labels=self.y_t, predictions=self.y_p)


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr=1e-3
        self.lr_decay_step=400
        self.lr_decay_rate=0.95

        self.learning_rate = tf.train.exponential_decay(
            self.lr,  # Base learning rate.
            self.global_step,  # Current index into the dataset.
            self.lr_decay_step,  # Decay step.
            self.lr_decay_rate,  # Decay rate.
            staircase=True
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            try:
                self.train_step = optimizer.minimize(self.cross_entropy, global_step=self.global_step)
            except Exception as e:
                print(e)

    def train(self):
        self.build()

        #with tf.Session() as sess:
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.iteration_error=[]
        for epoch_i in range(self.train_epoch):
            step_error=0
            batch_num=1
            for batch_i in iter_dataset(file_path,'train',self.batch_size):
                self.train_step.run(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.drop_prob: 0.5,self.IsTraining:True,self.drop_prob_dense:0.2})
                ce = self.cross_entropy.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.drop_prob: 0.5,self.IsTraining:True,self.drop_prob_dense:0.2})
                step_error+=ce

                if batch_num%10==0:
                    print('Epoch %d, batch %d, cross_entropy %g' % (epoch_i+1,batch_num, ce))
                batch_num+=1
            self.iteration_error.append(step_error)
        print('Cross Entropy Change:',self.iteration_error)
        #get training accuracy
        y_pred_training=[]
        y_true_training=[]
        for batch_i in iter_dataset(file_path=file_path,model='train',batch_size=self.batch_size,depth_image=self.depth_image):
            y_pred_training+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.drop_prob: 0.5,self.IsTraining:True,self.drop_prob_dense:0.2}))
            y_true_training+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.drop_prob: 0.5,self.IsTraining:True,self.drop_prob_dense:0.2}))
        #calculate accuracy
        p_correct=0
        for ii in range(len(y_pred_training)):
            if y_pred_training[ii]==y_true_training[ii]:
                p_correct+=1
        train_acc=p_correct/len(y_pred_training)
        print('Training Accuracy:',train_acc)

        #store the loss change
        with open('Loss_change_list.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.iteration_error, f, pickle.HIGHEST_PROTOCOL)


    def test(self):
        y_prediction=[]
        y_true=[]
        for batch_i in iter_dataset(file_path,'test',self.batch_size):
            #self.train_step.run(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.drop_prob: 0.5,self.IsTraining:False,self.drop_prob_dense:0.2})
            y_prediction += list(self.y_p.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.drop_prob: 0.5,self.IsTraining:False,self.drop_prob_dense:0.2}))
            y_true += list(self.y_t.eval(session=self.sess,feed_dict={self.x: batch_i[0],self.y_: batch_i[1], self.drop_prob: 0.5,self.IsTraining:False,self.drop_prob_dense:0.2}))
        print('Prediction:',y_prediction)
        print('True:',y_true)
        #calculate accuracy
        p_correct=0
        for ii in range(len(y_prediction)):
            if y_prediction[ii]==y_true[ii]:
                p_correct+=1
        acc=p_correct/len(y_prediction)
        print('Accuracy:',acc)

    def show_loss_change(self):
        plt.figure()
        x_axle=[(a+1) for a in range(len(self.iteration_error))]
        plt.plot(x_axle, self.iteration_error,linewidth=2)
        plt.title('Loss change of RGB images ', fontsize=20)
        plt.xlabel('Epoch Time', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.show()



if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    file_path='./data/'

    Model=CNNModel(file_path,20)
    #Model.build()
    Model.train()
    Model.test()
    Model.show_loss_change()