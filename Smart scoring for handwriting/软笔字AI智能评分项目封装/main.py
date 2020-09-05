# !/usr/bin/python
# -*- coding: utf-8 -*-


import segmentation_both
import binary
import os 
import retange_word
import similary
import transparency
import traceback
from json import dumps
from flask import Flask, request

app = web = Flask(__name__)
 
def RESTful(fn):
    def wrapper(*args, **kwargs):
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        try:
            return dumps({
                'success': True,
                'message': '',
                'data': fn(*args, **kwargs)
            }), 200, headers
        except Exception as e:
            return dumps({
                'success': False,
                'message': traceback.print_exc(),
                'data': None
            }), 400, headers
    wrapper.__name__ = fn.__name__
    return wrapper
 

 
@web.route('/score/Seg', methods = ['POST'])
@RESTful
def Seg():
    data = request.json
    result = segmentation_both.run(data)
    return result

 
@web.route('/score/retangeWord', methods = ['POST'])
@RESTful
def retangeWord():
    result = retange_word.run(request.json)
    return result
 
@web.route('/score/binary', methods = ['POST'])
@RESTful
def binarys():
    data = request.json
    image = data['image']
    return binary.run(image)
 
@web.route('/score/similary', methods = ['POST'])
@RESTful
def similarys():
    similaryResult = similary.run(request.json)
    return similaryResult
 
@web.route('/score/transparency', methods = ['POST'])
@RESTful
def trans():
    result = transparency.run(request.json)
    return result

 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000);