#-*-coding:utf-8 -*-

import tensorflow as tf

class TextCNN:
    def __init__(self,
                 sentence_length,
                 vocab_size,
                 num_classes,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda):
        #set params
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="drop_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        #init weights
        self.instantiate_weights()
        self.words_id = tf.contrib.lookup.index_table_from_tensor(mapping=self.vocab, default_value=0, name="words_id")

    def instantiate_weights(self):
        """
        define weights here
        :return:
        """
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        #with tf.variable_scope("embedding"):
            self.vocab = tf.get_variable("vocab", shape=[self.vocab_size], dtype=tf.string)
            self.words_embedding = tf.get_variable("words_embedding", shape=[self.vocab_size, self.embedding_size], dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.project_w = tf.get_variable("project_w", shape=[self.num_filters * len(self.filter_sizes), self.num_classes], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.project_b = tf.get_variable("project_b", shape=[self.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

    def inference(self, input_x):
        """

        :param input_x:
        :return:
        """
        input_ids = self.words_id.lookup(input_x)
        return self.online_inference(input_ids)

    def online_inference(self, input_ids):
        """

        :param input_ids:
        :return:
        """
        embedded_words = tf.nn.embedding_lookup(self.words_embedding, input_ids)#shape:[None,sentence_length,embedding_size]
        embedded_words_expand = tf.expand_dims(embedded_words, -1)#shape:[None,sentence_length,embedding_size,1]
        #conv --> relu --> max_pooling --> drop_out -->full connect
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=tf.AUTO_REUSE):
            #with tf.variable_scope("conv-maxpool-%s" % filter_size):
                #conv layer
                layer_w = tf.get_variable("layer_w", shape=[filter_size, self.embedding_size, 1, self.num_filters], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv = tf.nn.conv2d(
                    embedded_words_expand,
                    layer_w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                #batch normalization
                #conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training)
                #relu
                b = tf.get_variable("relu_b", initializer=tf.constant_initializer(0.1), shape=[self.num_filters], dtype=tf.float32)
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                #pooling layer
                pool = tf.nn.max_pool(h, ksize=[1, self.sentence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                      padding="VALID", name="pooled")#shape:[batch_size,sequence_length-filter_size+1,1,num_filters]
                #concat
                pooled_outputs.append(pool)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, shape=[-1, self.num_filters * len(self.filter_sizes)])

        #drop_out layers
        with tf.variable_scope("dropout"):
            drop_out = tf.nn.dropout(h_pool_flat, keep_prob=self.dropout_keep_prob)#shape:[batch_size,self.num_filters*len(self.fiter_sizes)]

        #output
        with tf.variable_scope("output"):
            logits = tf.matmul(drop_out, self.project_w) + self.project_b#shape:[batch_size, self.num_calsses]

        return logits

    def loss(self, logits, input_y):
        """

        :param logits:
        :param input_y:
        :return:
        """
        #l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_reg_lambda
        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(self.project_w)
        l2_loss += tf.nn.l2_loss(self.project_b)

        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
            loss = tf.reduce_mean(losses) + l2_loss * self.l2_reg_lambda

        return loss

    def accuracy(self, logits, input_y):
        """

        :param logits:
        :param input_y:
        :return:
        """
        prediction = tf.argmax(logits, 1, name="prediction")
        correct_prediction = tf.equal(prediction, tf.argmax(input_y, 1))
        with tf.variable_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"), name="accuracy")

        return accuracy

    def prediciton(self, logits):
        """

        :param logits:
        :return:
        """
        return tf.arg_max(logits, 1)

    def signature(self):
        """

        :return:
        """
        self.dropout_keep_prob = 1.0
        self.is_training = False
        x = tf.placeholder(tf.int32, shape=[None, self.sentence_length])
        scores = self.online_inference(x)
        predictions = self.prediciton(logits=scores)

        inputs = {"input_x": tf.saved_model.utils.build_tensor_info(x)}
        outputs = {"scores": tf.saved_model.utils.build_tensor_info(scores),
                    "predictions": tf.saved_model.utils.build_tensor_info(predictions)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs,
                                                                           outputs,
                                                                           method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)

        return signature