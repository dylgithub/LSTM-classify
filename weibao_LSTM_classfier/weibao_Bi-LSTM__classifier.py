#encoding=utf-8
import tensorflow as tf
import numpy as np
import time
from weibao_LSTM_classfier import weibo_data_helper
# 模型的超参数
tf.flags.DEFINE_integer("vector_size", 80, "每个单词向量的维度")
tf.flags.DEFINE_integer("sentence_length", 200, "设定的句子最大长度")
tf.flags.DEFINE_integer("n_hidden", 200, "隐藏层细胞的个数")
tf.flags.DEFINE_integer("num_classes", 3247, "类别种类数")
# 训练参数
tf.flags.DEFINE_integer("batch_size", 256, "每个批次的大小")
tf.flags.DEFINE_integer("num_epochs", 1000, "训练的轮数")
tf.flags.DEFINE_float("init_learning_rate", 0.02, "初始学习率")
FLAGS = tf.flags.FLAGS
start_time=time.time()
x=tf.placeholder("float",[None,FLAGS.sentence_length,FLAGS.vector_size]) #输入的x变量是三维的第一维是批次大小，第二，第三维组成一个句子
y=tf.placeholder("float",[None,FLAGS.num_classes])
#动态双向GRU网络的构建
lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(FLAGS.n_hidden,forget_bias=1.0) #创建正向的cell
lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(FLAGS.n_hidden,forget_bias=1.0) #创建反向的cell
outputs,outputs_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
outputs=tf.concat(outputs,2)
outputs=tf.transpose(outputs,[1,0,2])
pred=tf.contrib.layers.fully_connected(outputs[-1],FLAGS.num_classes,activation_fn=None)

#计算损失值选择优化器
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.init_learning_rate).minimize(cost)

#用精度评估模型
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    vec_list,lables = weibo_data_helper.get_vec_data(FLAGS.sentence_length,FLAGS.vector_size)
    vec_list=np.array(vec_list)
    labels=np.array(lables)
    # test_vec_list=vec_list[-5000:]
    # test_labels=labels[-5000:]
    # vec_list=vec_list[:-5000]
    # labels=labels[:-5000]
    #批量获得数据
    num_inter=int(len(labels)/FLAGS.batch_size)
    for j in range(FLAGS.num_epochs):
        for i in range(num_inter):
            start=i*FLAGS.batch_size
            end=(i+1)*FLAGS.batch_size
            sess.run(optimizer, feed_dict={x:vec_list[start:end], y: labels[start:end]})
            if i%20==0:
                train_accuracy=accuracy.eval(feed_dict={x:vec_list[start:end],y:labels[start:end]})
                print("Epoch %d:Step %d accuracy is %f" % (j,i,train_accuracy))
    # test_accuracy = accuracy.eval(feed_dict={x: test_vec_list, y: test_labels})
    # print("test accuracy is %f" % test_accuracy)
    end_time=time.time()
    print('总共花费：',end_time-start_time)