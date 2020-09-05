
from __future__ import absolute_import, division, print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
import tensorflow as tf
from six.moves import urllib



# model_file = "/var/score/py-script/Standard_words/freezed.pb"
model_file="D:/item/output/py-script-v4/Standard_words/freezed.pb"
with open(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

session = tf.compat.v1.Session()
tensor = session.graph.get_tensor_by_name('final_probs:0')    

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_path=None):
        if not label_path:
            tf.logging.fatal('please specify the label file.')
            return
        self.node_lookup = self.load(label_path)

    def load(self, label_path):
        """Loads a human readable English name for each softmax node.
        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.
        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(label_path):
            tf.logging.fatal('File does not exist %s', label_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
        id_to_human = {}
        for line in proto_as_ascii_lines:
            if line.find(':') < 0:
                continue
            _id, human = line.rstrip('\n').split(':')
            id_to_human[int(_id)] = human

        return id_to_human

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def run_inference_on_image(image,freezedLablePath):
    """Runs inference on an image.
    Args:
      image: Image file name.
    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = open(image, 'rb').read()



    predictions = session.run(tensor,
                               {'input:0': image_data})
    predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
    node_lookup = NodeLookup(freezedLablePath)

    top_k = predictions.argsort()[-3:][::-1]
    top_names = []
    top_scores=[]
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        top_names.append(human_string)
        score = predictions[node_id]
        top_scores.append(score)

    return top_names,top_scores


def run(freezedLablePath,imagePath,standardWordsPath):
    names,scores=run_inference_on_image(imagePath,freezedLablePath)
    
    dict={}
    num=[]
    paths=[]
    dict["img_path"]=imagePath
    if names[0][4:] == "165":
        dict["whether_recognize"] = 1 
        dict["whether_space"]=1
        
    elif scores[0]<0.35:
        dict["whether_recognize"] = 0
    else:
        
        dict["whether_recognize"] = 1        
        
        dict["whether_space"]=0
        for i in range(1):
            new_name="img"+names[i][4:]+".jpg"
            path=os.path.join(standardWordsPath,new_name)
            paths.append(path)
            num.append("img"+names[i][4:])
        dict["recog_img"]=num    
        dict["results"]=paths
            
    return dict


