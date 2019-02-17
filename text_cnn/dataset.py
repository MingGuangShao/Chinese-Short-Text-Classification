# -*- coding: utf-8 -*-
import tensorflow as tf
from utils import sentence_padding

def load_word_vector(file_name, vocab_size, embedding_size, field_delim, words, embeddings):
    """
    load word embedding data , split it to words and embeddings, assign it to tf.variable
    :param file_name: word_embedding file path(str)
    :param vocab_size: num of words in word_embedding file
    :param embedding_size: embedding size of words
    :param field_delim: field delimiter in word_embedding file
    :param words: words assign (tf.variable)
    :param embeddings: embeddings assign (tf.variable)
    :return: variable_words, variable_embeddings
    """
    #generate a file queue
    file_queue = tf.train.string_input_producer([file_name], shuffle=True, name="word_load_queue")
    #define reader
    reader = tf.TextLineReader()
    keys, values = reader.read_up_to(queue=file_queue, num_records=vocab_size)
    #split and padding embedding
    sliced = sentence_padding(values=values, field_delim=field_delim, max_sentence_len=embedding_size+1, pad_value='0.0')
    #split words and  values
    words_, embeddings_ = tf.split(value=sliced, num_or_size_splits=[1, embedding_size], axis=1)
    words_ = tf.reshape(words_, [vocab_size],name="wocab_vectors")
    embeddings_ = tf.string_to_number(embeddings_, name='embeddings_vectors_number')
    #assign variable
    init_words_var = tf.assign(words, words_)
    init_embeddings_var = tf.assign(embeddings,embeddings_)

    return init_words_var, init_embeddings_var

def load_train_words_and_lables(reader, file_queue, max_sentence_len, batch_size, num_classes, field_delim, word_delim):
    """
    load batch of train data, split batch data to words and labels
    :param reader: TextLineReader
    :param file_queue: file_queue of train data path
    :param max_sentence_len: max sentence length
    :param batch_size: batch size defined by user
    :param num_classes: classified classes defined by user
    :param field_delim: delimiter between sentence and labels
    :param word_delim: word delimiter in sentence
    :return: words and one-hot labels
    """
    keys, values = reader.read_up_to(queue=file_queue, num_records=batch_size)
    #decode
    sentences, labels = tf.decode_csv(values, [[''], [1]], field_delim=field_delim)
    #split and padding, pad_value is the first word in words_embedding
    words = sentence_padding(values=sentences, field_delim=word_delim, max_sentence_len=max_sentence_len, pad_value='</s>')
    one_hot_labels = tf.one_hot(labels, num_classes)
    return words, one_hot_labels

def load_valid_words_and_labels(file_name, valid_size, max_sentence_len, num_classes, field_delim, word_delim):
    """
    load valid data , spit data to words and labels
    :param file_name: valid(eval,test)d file path
    :param valid_size: samples of valid data
    :param max_sentence_len: max sentence length defined by user
    :param num_classes: classified classes defined by user
    :param field_delim: delimiter between sentence and labels
    :param word_delim: word delimiter in sentence
    :return: words and one_hot_labels
    """
    #generate a file queue
    file_queue = tf.train.string_input_producer([file_name], shuffle=True, name="valid_load_queue")
    #define reader
    reader = tf.TextLineReader()
    keys, values = reader.read_up_to(queue=file_queue, num_records=valid_size)
    #decode
    sentences, labels = tf.decode_csv(values, [[''], [1]], field_delim=field_delim)
    #split and padding, pad value is the first word in words_embedding
    words = sentence_padding(values=sentences, field_delim=word_delim, max_sentence_len=max_sentence_len, pad_value='</s>')
    one_hot_labels = tf.one_hot(labels, num_classes)
    return words, one_hot_labels

'''
#unit test
with tf.Graph().as_default():
    with tf.Session() as sess:
        #construct graph
        train_file_queue = tf.train.string_input_producer(["data.csv"], shuffle=True, name="train_load_queue")
        train_reader = tf.TextLineReader()
        values = load_train_words_and_lables(reader=train_reader, file_queue=train_file_queue, max_sentence_len=5, batch_size=5, num_classes=2,
                                             field_delim=',', word_delim=' ')
        #session-run
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print(sess.run(values))
        coord.request_stop()
        coord.join(threads)
'''