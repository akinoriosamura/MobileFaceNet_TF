# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.data_process import load_data
from verification import evaluate
from scipy.optimize import brentq
from scipy import interpolate
from nets.MobileFaceNet import inference
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import argparse
import time
import sys
import re
import os


def create_save_model(args, model_dir, graph, sess):
    # check data type
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(i)   # i.name if you want just a name
    print("finish check")
    # save graphdef file to pb
    print("Save frozen graph")
    graphdef_n = "facenet.pb"
    graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ["embeddings"])
    tf.train.write_graph(graph_def, model_dir, graphdef_n, as_text=False)

    # save SavedModel
    print("get tensor")
    inputs_placeholder = tf.get_default_graph().get_tensor_by_name("img_inputs:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

    print("start save saved_model")
    save_model_dir = os.path.join(model_dir, "SavedModel")
    builder = tf.saved_model.builder.SavedModelBuilder(save_model_dir)
    signature = tf.saved_model.predict_signature_def(
        {"img_inputs": inputs_placeholder}, outputs={"embeddings": embeddings}
    )

    # using custom tag instead of: tags=[tf.saved_model.tag_constants.SERVING]
    builder.add_meta_graph_and_variables(
        sess=sess, tags=[
            tf.saved_model.tag_constants.SERVING], signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()
    print("finish save saved_model")

    # save tflite model
    converter = tf.lite.TFLiteConverter.from_saved_model(save_model_dir)
    tflite_model = converter.convert()
    with open(os.path.join(model_dir, "facenet.tflite"), 'wb') as f:
        f.write(tflite_model)
    print("finish save tflite")


def main(args):
    with tf.Graph().as_default() as inf_g:
        # Get input and output tensors, ignore phase_train_placeholder for it have default value.
        inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
        phase_train_placeholder = tf.constant(False, name='phase_train')

        # identity the input, for inference
        # inputs = tf.identity(inputs, 'input')
        prelogits, net_points = inference(inputs, bottleneck_layer_size=args.embedding_size, phase_train=phase_train_placeholder)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        save_params = tf.trainable_variables()
        saver = tf.train.Saver(save_params, max_to_keep=None)
        sess = tf.Session(
            graph=inf_g,
            config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        with sess.as_default():
            # Load the model
            print('Model directory: {}'.format(args.model))
            ckpt = tf.train.get_checkpoint_state(args.model)
            print('ckpt: {}'.format(ckpt))
            model_path = ckpt.model_checkpoint_path
            assert (ckpt and model_path)
            print('Checkpoint file: {}'.format(model_path))
            import pdb; pdb.set_trace()
            saver.restore(sess, model_path)

            create_save_model(args, args.model, inf_g, sess)


def parse_arguments(argv):
    '''test parameters'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='./arch/pretrained_model/')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=100)
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=192)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))