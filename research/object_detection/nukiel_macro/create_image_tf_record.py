# -*- coding: utf-8 -*-

# 将训练的图片转化为tf.record
# 基于create_pet_tf_record.py修改
# nukiel 2018/4/29

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

import sys  

research_root='/home/leikun/models/research/'
sys.path.insert(0,research_root)

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/leikun/temp/AI100-W9', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/home/leikun/temp/AI100-W9/out', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '/home/leikun/temp/AI100-W9/labels_items.txt','Path to label map proto')
FLAGS = flags.FLAGS


def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    example: The converted tf.Example.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  # 获得.jpg的文件路径,data['filename']为具体的图像名
  img_path = os.path.join(image_subdirectory, data['filename'])

  with tf.gfile.GFile(img_path,'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  # 获得图像的的宽和高
  # <size>
  # <width>1152</width>
  # <height>864</height>
  # <depth>3</depth>
  # </size>
  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  # 获得所有目标物品的边界定义等信息
  # <object>
  # <name>computer</name>
  # <pose>Unspecified</pose>
  # <truncated>0</truncated>
  # <difficult>0</difficult>
  # <bndbox>
  # <xmin>462</xmin><ymin>493</ymin><xmax>604</xmax><ymax>727</ymax>
  # </bndbox>
  # </object>

  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)

    # 图片命名格式不是例子给的'american_pit_bull_terrier_105.jpg'
    # class_name没办法通过get_class_name_from_filename获得,
    # 直接采用xml中的name属性赋值
    # 此处是坑
    
    class_name =obj['name'] # get_class_name_from_filename(data['filename'])
    classes_text.append(class_name.encode('utf8'))

    classes.append(label_map_dict[class_name])

    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'])
    
    # 取掉扩展名
    image_name=os.path.splitext(data['filename'])[0]
    
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_name), #此处有修改
      'image/source_id': dataset_util.bytes_feature(image_name), #此处有修改
      'image/key/sha256': dataset_util.bytes_feature(key),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))
    # .xml文件路径 /home/leikun/temp/AI100-W9/annotations/xmls/名字.xml
    path = os.path.join(annotations_dir, 'xmls', example + '.xml')

    if not os.path.exists(path):
      logging.warning('Could not find %s, ignoring example.', path)
      continue
    
    # 在python 3.x运行,一定要用'rb',而不是'r'
    # 否则报错:UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
    # 此处是坑!
    with tf.gfile.GFile(path, 'rb') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
    writer.write(tf_example.SerializeToString())

  writer.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):

  # 数据文件夹的存放路径,/home/leikun/temp/AI100-W9
  data_dir ='/home/leikun/temp/AI100-W9' # FLAGS.data_dir

  # label_items.txt文件路径/home/leikun/temp/AI100-W9/labels_items.txt
  label_path = '/home/leikun/temp/AI100-W9/labels_items.txt'
  label_map_dict = label_map_util.get_label_map_dict(label_path) # (FLAGS.label_map_path)

  logging.info('Reading from Pet dataset.')
  # 在data_dir下有:
  # images文件夹 : 保存所有图片信息 .jpg格式
  # annotations文件夹:保存所有的.xml格式的文件夹xlms
  image_dir = os.path.join(data_dir, 'images')
  annotations_dir = os.path.join(data_dir, 'annotations')
  examples_path = os.path.join(annotations_dir, 'trainval.txt')

  # 获得训练数据的文件名列表
  examples_list = dataset_util.read_examples_list(examples_path)

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.

  # 随机分布
  random.seed(42)
  random.shuffle(examples_list)

  # 取70%作为训练集,30%作为测试集
  num_examples = len(examples_list)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  
  print('%d training and %d validation examples.',
               len(train_examples), len(val_examples))
  
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  # tf.record数据的输出路径
  output_dir = '/home/leikun/temp/AI100-W9/out'
  train_output_path = os.path.join(output_dir, 'pet_train.record') # (FLAGS.output_dir, 'pet_train.record')
  val_output_path = os.path.join(output_dir, 'pet_val.record') # (FLAGS.output_dir, 'pet_val.record')

  # 生成训练集数据
  create_tf_record(train_output_path, label_map_dict, annotations_dir,image_dir, train_examples)

  # 生成测试集数据
  create_tf_record(val_output_path, label_map_dict, annotations_dir,image_dir, val_examples)

if __name__ == '__main__':
  tf.app.run()

# 这里有个大坑!
# 一定要在python 2.7的环境中执行
# 我用的是vs code 运行的,setting.json文件中已经指定了py27下的python2.7了
# 但是没有激活py27,如下:
# leikun@Z240:~/models/research$ conda env list
# # conda environments:
# #
# base                  *  /home/leikun/anaconda3
# py27                     /home/leikun/anaconda3/envs/py27
# 后来激活后,就能运行了!
# source activate py27

# 可能前面的坑也是因为是在3的环境中的原因....

# python object_detection/nukiel_macro/create_image_tf_record.py --label_map_path=/home/leikun/temp/AI100-W9/labels_items.txt --data_dir=/home/leikun/temp/AI100-W9 --output_dir=/home/leikun/temp/AI100-W9/out