# -*- coding: utf-8 -*-

from grpc.beta import implementations
import numpy
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', '0.0.0.0:8500', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

def online_inference(hostport, data):
    ""
    #creat connection
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()

    request.model_spec.name = "text_cnn"
    request.model_spec.signature_name = "serving_default"

    request.inputs["input_x"].ParseFromString(tf.contrib.util.make_tensor_proto(
        data,
        dtype = tf.int32,
        shape = [1,25]).SerializeToString()
    )
    '''
    request.inputs["input"].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            data,
            dtype = tf.int32,
            shape = [1,25]
        )
    )
    '''
    #predict
    response = stub.Predict(request, 5.0)
    result = {}
    for key in response.outputs:
        tensor_proto = response.outputs[key]
        nd_array = tf.contrib.util.make_ndarray(tensor_proto)
        result[key] = nd_array

    return result

def main(_):
    if not FLAGS.server:
        print("please specify server host:port")
    #data = [['你','好','请问','我的','电脑','哪儿',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']]
    data = [[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    #data_input = [i.encode('utf-8') for i in data]
    result = online_inference(FLAGS.server, data)

    print("result: ")
    print(result)

if __name__ == '__main__':
    tf.app.run()