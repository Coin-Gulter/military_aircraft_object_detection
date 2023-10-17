"""Hparams for model architecture and trainer."""
import ast
from collections import abc
import copy
from typing import Any, Dict, Text
import six
import tensorflow as tf
import yaml


def eval_str_fn(val):
    """
    Function to evaluate a string expression.

    Args:
        val (str): String expression to be evaluated.

    Returns:
        Any: Evaluated value of the expression.
    """

    # Check if the string is a boolean literal.
    if val in {'true', 'false'}:
        return val == 'true'

    # Try to evaluate the string as a Python expression.
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # If the expression is invalid, return the original string.
        return val
    

class Config(object):
  """A config utility class."""

  def __init__(self, config_dict=None):
    self.update(config_dict)

  def __setattr__(self, k, v):
    self.__dict__[k] = Config(v) if isinstance(v, dict) else copy.deepcopy(v)

  def __getattr__(self, k):
    return self.__dict__[k]

  def __getitem__(self, k):
    return self.__dict__[k]

  def __repr__(self):
    return repr(self.as_dict())

  def __deepcopy__(self, memodict):
    return type(self)(self.as_dict())

  def __str__(self):
    try:
      return yaml.dump(self.as_dict(), indent=4)
    except TypeError:
      return str(self.as_dict())

  def _update(self, config_dict, allow_new_keys=True):
    """Recursively update internal members."""
    if not config_dict:
      return

    for k, v in six.iteritems(config_dict):
      if k not in self.__dict__:
        if allow_new_keys:
          self.__setattr__(k, v)
        else:
          raise KeyError('Key `{}` does not exist for overriding. '.format(k))
      else:
        if isinstance(self.__dict__[k], Config) and isinstance(v, dict):
          self.__dict__[k]._update(v, allow_new_keys)
        elif isinstance(self.__dict__[k], Config) and isinstance(v, Config):
          self.__dict__[k]._update(v.as_dict(), allow_new_keys)
        else:
          self.__setattr__(k, v)

  def get(self, k, default_value=None):
    return self.__dict__.get(k, default_value)

  def update(self, config_dict):
    """Update members while allowing new keys."""
    self._update(config_dict, allow_new_keys=True)

  def keys(self):
    return self.__dict__.keys()

  def override(self, config_dict_or_str, allow_new_keys=False):
    """Update members while disallowing new keys."""
    if isinstance(config_dict_or_str, str):
      if not config_dict_or_str:
        return
      elif '=' in config_dict_or_str:
        config_dict = self.parse_from_str(config_dict_or_str)
      elif config_dict_or_str.endswith('.yaml'):
        config_dict = self.parse_from_yaml(config_dict_or_str)
      else:
        raise ValueError(
            'Invalid string {}, must end with .yaml or contains "=".'.format(
                config_dict_or_str))
    elif isinstance(config_dict_or_str, dict):
      config_dict = config_dict_or_str
    else:
      raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

    self._update(config_dict, allow_new_keys)

  def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
    """Parses a yaml file and returns a dictionary."""
    with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
      config_dict = yaml.load(f, Loader=yaml.FullLoader)
      return config_dict

  def save_to_yaml(self, yaml_file_path):
    """Write a dictionary into a yaml file."""
    with tf.io.gfile.GFile(yaml_file_path, 'w') as f:
      yaml.dump(self.as_dict(), f, default_flow_style=False)

  def parse_from_str(self, config_str: Text) -> Dict[Any, Any]:
    """Parse a string like 'x.y=1,x.z=2' to nested dict {x: {y: 1, z: 2}}."""
    if not config_str:
      return {}
    config_dict = {}
    try:
      for kv_pair in config_str.split(','):
        if not kv_pair:  # skip empty string
          continue
        key_str, value_str = kv_pair.split('=')
        key_str = key_str.strip()

        def add_kv_recursive(k, v):
          """Recursively parse x.y.z=tt to {x: {y: {z: tt}}}."""
          if '.' not in k:
            if '*' in v:
              # we reserve * to split arrays.
              return {k: [eval_str_fn(vv) for vv in v.split('*')]}
            return {k: eval_str_fn(v)}
          pos = k.index('.')
          return {k[:pos]: add_kv_recursive(k[pos + 1:], v)}

        def merge_dict_recursive(target, src):
          """Recursively merge two nested dictionary."""
          for k in src.keys():
            if ((k in target and isinstance(target[k], dict) and
                 isinstance(src[k], abc.Mapping))):
              merge_dict_recursive(target[k], src[k])
            else:
              target[k] = src[k]

        merge_dict_recursive(config_dict, add_kv_recursive(key_str, value_str))
      return config_dict
    except ValueError:
      raise ValueError('Invalid config_str: {}'.format(config_str))

  def as_dict(self):
    """Returns a dict representation."""
    config_dict = {}
    for k, v in six.iteritems(self.__dict__):
      if isinstance(v, Config):
        config_dict[k] = v.as_dict()
      else:
        config_dict[k] = copy.deepcopy(v)
    return config_dict
    # pylint: enable=protected-access


def default_configs():
    """Returns a default detection configs."""
    h = Config()

    # model.
    h.name = 'military_aircraft'
    h.pretrained = False
    h.save_model_name = 'military_aircraft_bbox.keras'
    h.save_model_folder = './'

    # activation type: see activation_fn in utils.py.
    h.act_type = 'softmax'

    # input preprocessing parameters
    h.image_size = 360  # An integer or a string WxH such as 360x360.
    h.batch_size = 64

    h.image_classes = [{'A20':0,'A1':1,'A2':2,'A3':3,'A4':4,'A5':5,'A6':6,'A7':7,'A8':8,'A9':9,'A10':10,
                       'A11':11,'A12':12,'A13':13,'A14':14,'A15':15,'A16':16,'A17':17,'A18':18,'A19':19}]

    # dataset specific parameters
    h.num_classes = 20   # actual classes.
    h.regresion_number = '4'
    h.num_objects = 50
    h.train_file = 'dataset/ImageSets/Main/train.txt'
    h.test_file = 'dataset/ImageSets/Main/test.txt'
    h.data_annotation_path = 'dataset/Annotations/Horizontal Bounding Boxes/'
    h.train_img_path = 'dataset/JPEGImages/'
    h.test_img_path = 'dataset/JPEGImages/'
    h.with_test = True # True if want test model after traing else False
    h.validation_split = 0.1

    # optimization
    h.optimizer = 'adam'  # can be 'adam' or 'sgd'.
    h.learning_rate = 0.0001  # 0.008 for adam.
    h.lr_dropout = 0.3
    h.num_epochs = 6
    h.metrics = [{'classifier_head' : 'accuracy', 'regressor_head' : 'mse' }]

    # loss
    h.loss = [{'classifier_head' : 'categorical_crossentropy', 'regressor_head' : 'mse' }] 

    return h


page_model_param_dict = {
    'military_aircraft':
        dict(
            image_size = 460,
            num_epochs = 8,
            pretrained = False,
            learning_rate = 0.006,
            with_test = True,
            validation_split = 0.1,
            test_file = 'dataset/ImageSets/Main/test.txt',
            train_file = 'dataset/ImageSets/Main/train.txt',
            data_annotation_path = 'dataset/Annotations/Horizontal Bounding Boxes/',
            train_img_path = 'dataset/JPEGImages/',
            test_img_path = 'dataset/JPEGImages/',
            batch_size = 64,
        )
}


def get_config(model_name='military_aircraft'):
  """Get the default config"""
  h = default_configs()
  if model_name in page_model_param_dict:
    h.override(page_model_param_dict[model_name])
  else:
    raise ValueError('Unknown model name: {}'.format(model_name))

  return h
