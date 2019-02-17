# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from model_cnn import TextCNN
from data_loader import load_word_vector, load_train_words_and_lables, load_valid_words_and_labels
import os,time,datetime
from utils import export_online, make_summary

#parameters
# Data loading params
tf.app.flags.DEFINE_string("train_cols","content,int_label","tables columns for training")
tf.app.flags.DEFINE_string("field_delim",",","delimiter of record fields")
tf.app.flags.DEFINE_string("word_delim"," ","delimiter of word fields")
tf.app.flags.DEFINE_integer("word_size",1450,"num of embedding words")
tf.app.flags.DEFINE_integer("eval_size",3965,"num of samples in eval data")

#path params
tf.app.flags.DEFINE_string("train_data_path","train.csv","path of training data")
tf.app.flags.DEFINE_string("word_embedding_path","embedding.csv","path of word embedding")
tf.app.flags.DEFINE_string("summary_dir","./summary/","directory for summary log")
tf.app.flags.DEFINE_string("checkpoint_dir","./check_point/","path of saving model")
tf.app.flags.DEFINE_string("export_online_path","./export_online/","path of saving online model")
tf.app.flags.DEFINE_string("predict_data_path","test.csv","path of predict path")
tf.app.flags.DEFINE_string("predict_result","predict_result","dir of prediction result")

#model param
tf.app.flags.DEFINE_integer("embedding_size",100,"embedding size")
tf.app.flags.DEFINE_integer("sentence_length",25,"length of sentence")
tf.app.flags.DEFINE_integer("eval_every",500,"eval every epoches")
tf.app.flags.DEFINE_integer("num_class",2,"number of classes(default:2)")
tf.app.flags.DEFINE_string("filter_size","3,4,5","filters size")
tf.app.flags.DEFINE_integer("num_filters",128,"num pf filters per filter size")
tf.app.flags.DEFINE_integer("batch_size",64,"batch_szie")
tf.app.flags.DEFINE_integer("num_epochs",1000,"number of epochs to run")
tf.app.flags.DEFINE_integer("decay_steps",1000,"how many steps before decay learning rate")
tf.app.flags.DEFINE_integer("num_checkpoints",5,"number of checkpoints to store")

tf.app.flags.DEFINE_float("drop_keep_prob",0.5,"dropout keep probability")
tf.app.flags.DEFINE_float("decay_rate",1.0,"rate of decay for learning rate")

#predict and online export params
tf.app.flags.DEFINE_string("predict_cols","content,int_label","columns for prediction")
tf.app.flags.DEFINE_string("online_model","yes","online model:yes or no")

FLAGS=tf.app.flags.FLAGS

def train():
    with tf.Graph().as_default():
        #build model graph
        model = TextCNN(sentence_length=FLAGS.sentence_length,
                        vocab_size=FLAGS.word_size,
                        num_classes=FLAGS.num_class,
                        embedding_size=FLAGS.embedding_size,
                        filter_sizes=list(map(int, FLAGS.filter_size.split(","))),
                        num_filters=FLAGS.num_filters,
                        l2_reg_lambda=0.1)
        #construct data loader grpah
        init_words, init_embeddings = load_word_vector(file_name=FLAGS.word_embedding_path,
                                                       vocab_size=FLAGS.word_size,
                                                       embedding_size=FLAGS.embedding_size,
                                                       field_delim=FLAGS.field_delim,
                                                       words=model.vocab,
                                                       embeddings=model.words_embedding)

        train_file_queue = tf.train.string_input_producer([FLAGS.train_data_path], shuffle=True, name="train_load_queue")
        train_reader = tf.TextLineReader()
        train_words, train_labels = load_train_words_and_lables(reader=train_reader,
                                                                file_queue=train_file_queue,
                                                                max_sentence_len=FLAGS.sentence_length,
                                                                batch_size=FLAGS.batch_size,
                                                                num_classes=FLAGS.num_class,
                                                                field_delim=FLAGS.field_delim,
                                                                word_delim=FLAGS.word_delim)

        valid_words, valid_labels = load_valid_words_and_labels(file_name=FLAGS.predict_data_path,
                                                                valid_size=FLAGS.eval_size,
                                                                max_sentence_len=FLAGS.sentence_length,
                                                                num_classes=FLAGS.num_class,
                                                                field_delim=FLAGS.field_delim,
                                                                word_delim=FLAGS.word_delim)

        train_logits = model.inference(train_words)
        train_loss = model.loss(logits=train_logits, input_y=train_labels)
        train_acc = model.accuracy(logits=train_logits, input_y=train_labels)

        valid_logits = model.inference(valid_words)
        valid_loss = model.loss(logits=valid_logits, input_y=valid_labels)
        valid_acc = model.accuracy(logits=valid_logits, input_y=valid_labels)

        #construct saver
        saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

        #define training procedure, note:batchnorm
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=3e-4, global_step=global_step, decay_steps=FLAGS.decay_steps,
                                                   decay_rate=FLAGS.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(train_loss)
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)
        #gradients, variables = zip(*optimizer.compute_gradients(train_loss))
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        #    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        #train summary
        train_summary_op = make_summary(grads_and_vars, train_loss, train_acc)
        # session config
        #session_conf = tf.ConfigProto(allow_soft_placement=True,
        #                              log_device_placement=FLAGS.log_device_placement)
        session_conf = tf.ConfigProto()
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            #init words and embeddings
            sess.run([init_words, init_embeddings])
            #initialize hash table for looking up word
            tf.tables_initializer().run()

            #feed dict
            param_feed_train = {model.dropout_keep_prob: FLAGS.drop_keep_prob, model.is_training: True}
            param_feed_valid = {model.dropout_keep_prob: 1.0, model.is_training: False}

            timestamp = str(int(time.time()))
            train_summary_dir = os.path.join(FLAGS.summary_dir, "train_summaries", timestamp)
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            for i in range(FLAGS.num_epochs):
                _, _step, _summaries, _loss, _acc =sess.run([train_op, global_step, train_summary_op, train_loss, train_acc], feed_dict=param_feed_train)
                _time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(_time_str, _step, _loss, _acc))
                train_summary_writer.add_summary(_summaries, _step)

                if (i + 1) % FLAGS.eval_every == 0:
                    _valid_loss, _valid_acc = sess.run([valid_loss, valid_acc], feed_dict=param_feed_valid)
                    print("valid_loss {:g}, valid_accuracy {:g}".format(_valid_loss, _valid_acc))

            try:
                if FLAGS.online_model == "yes":
                    online_path = export_online(sess=sess, export_path=FLAGS.export_online_path, count=FLAGS.num_epochs, signature=model.signature())
                    print("Save online model {}\n".format(online_path))
                saver.save(sess=sess,save_path=FLAGS.checkpoint_dir, global_step=FLAGS.num_epochs)
            except tf.errors.OutOfRangeError:
                pass

            finally:
                coord.request_stop()
                coord.join(threads)

def main(argv=None):
    train()

if __name__ == "__main__":
    tf.app.run()