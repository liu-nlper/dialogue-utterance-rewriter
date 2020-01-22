#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/21 22:54
# @Author  : jxliu
# @Email   : jxliu.nlper@gmail.com
# @File    : params_mapping.py

import re
import collections
import tensorflow as tf


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def share_parameters(initialized_variable_names):
  """
  Args:
    initialized_variable_names: list(str)

  Returns:
    assignment_map: parameters mapping dict
  """
  # 写入配置文件
  num_layers = 6

  assignment_map = dict()

  # embddding layer
  key = 'bert/embeddings/word_embeddings'
  value = 'model/Transformer/embedding_shared_weights/' \
          'embedding_and_softmax/weights'
  assignment_map[key] = value
  initialized_variable_names[value] = 1
  initialized_variable_names[value + ":0"] = 1

  for i in range(num_layers):
    prefix_key = 'bert/encoder/layer_{0}'.format(i)
    prefix_value = 'model/Transformer/encoder_stack/layer_{0}'.format(i)
    # q, k, v
    key = '{0}/attention/self/query/kernel'.format(prefix_key)
    value = '{0}/self_attention/self_attention/q/kernel'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/attention/self/key/kernel'.format(prefix_key)
    value = '{0}/self_attention/self_attention/k/kernel'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/attention/self/value/kernel'.format(prefix_key)
    value = '{0}/self_attention/self_attention/v/kernel'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/attention/output/dense/kernel'.format(prefix_key)
    value = '{0}/self_attention/self_attention/' \
            'output_transform/kernel'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/attention/output/LayerNorm/gamma'.format(prefix_key)
    value = '{0}/self_attention/layer_normalization/' \
            'layer_norm_scale'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/attention/output/LayerNorm/beta'.format(prefix_key)
    value = '{0}/self_attention/layer_normalization/' \
            'layer_norm_bias'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/intermediate/dense/kernel'.format(prefix_key)
    value = '{0}/ffn/feed_foward_network/' \
            'filter_layer/kernel'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/intermediate/dense/bias'.format(prefix_key)
    value = '{0}/ffn/feed_foward_network/' \
            'filter_layer/bias'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/output/dense/kernel'.format(prefix_key)
    value = '{0}/ffn/feed_foward_network/' \
            'output_layer/kernel'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/output/dense/bias'.format(prefix_key)
    value = '{0}/ffn/feed_foward_network/' \
            'output_layer/bias'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/output/LayerNorm/gamma'.format(prefix_key)
    value = '{0}/ffn/layer_normalization/' \
            'layer_norm_scale'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1
    key = '{0}/output/LayerNorm/beta'.format(prefix_key)
    value = '{0}/ffn/layer_normalization/' \
            'layer_norm_bias'.format(prefix_value)
    assignment_map[key] = value
    initialized_variable_names[value] = 1
    initialized_variable_names[value + ":0"] = 1

  return assignment_map


def init_model_with_bert(init_checkoint):
  tvars = tf.trainable_variables()
  if init_checkoint:
    (assignment_map, initialized_variable_names
     ) = get_assignment_map_from_checkpoint(tvars, init_checkoint)

    # init encoder parameters with pre-trained BERT
    assignment_map = share_parameters(initialized_variable_names)
    tf.train.init_from_checkpoint(init_checkoint, assignment_map)

    tf.logging.info('**** Trainable Variables ****')
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)


if __name__ == '__main__':
  pass