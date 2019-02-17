import tensorflow as tf
import os

def sentence_padding(values, field_delim, max_sentence_len, pad_value):
    """
    padding sentence to max_sentence_length or slice sentence to max_sentence_length
    :param values: data
    :param field_delim: field delimiter
    :param max_sentence_len: max sentence length
    :param pad_value: padding value
    :return:
    """
    #split and padding, tf.string_split return a sparse tensor
    sparse_values = tf.string_split(values, delimiter=field_delim)
    #convert sparse tensor to dense tensor
    dense_values = tf.sparse_tensor_to_dense(sp_input=sparse_values, default_value=pad_value)
    #type of padding, [[0,0],[0,max_sentence_len]] means the start and end to each dimension.
    #[0,0] means start 0 and end 0 in dimension 1 of data
    padding = tf.constant([[0, 0], [0, max_sentence_len]])
    padded = tf.pad(tensor=dense_values, paddings=padding, mode='CONSTANT', constant_values=pad_value)
    #slice data to max sentence length, [0,0] means begin of dimension 1 and dimension 2, [-1, max_sentence_len]
    #means end of dimension 1 and dimension 2
    sliced = tf.slice(padded, [0, 0], [-1, max_sentence_len])

    return sliced

def export_online(sess, export_path, count, signature):
    """
    export(save) model online
    :param sess: tf.Session()
    :param export_path: path to save online model
    :param count: num of epoch
    :param signature: defined for builder
    :return: online model path
    """
    export_online_path = os.path.join(tf.compat.as_bytes(export_path), tf.compat.as_bytes(str(count)))
    print("Exportping train model online to", export_online_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_online_path)
    builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()
    print("save model online done!")

    return export_online_path

def make_summary(grads_and_vars, loss, accuracy):
    """
    make summary for tensorboard
    :param grads_and_vars: grads and vars
    :param loss:
    :param accuracy:
    :return: tensorflow op
    """
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar("loss", loss)
    acc_summary = tf.summary.scalar("accuracy", accuracy)

    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])

    return train_summary_op