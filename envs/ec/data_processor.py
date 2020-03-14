import time
import os
import json
import numpy as np
'''
保存的数据格式为 list dict
'''
class DataSaver:
    def __init__(self,func_name):
        dir = os.path.join(os.getcwd(), "data", func_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = func_name+"_"+now
        self.file_dir = os.path.join(dir,self.filename)
        self.data = {}


    def add_item(self,key,item):
        #添加一项数据(list),或者单个数据
        #如果是数组，则化为list
        if type(item) == np.ndarray:
            item = item.tolist()
        self.data[key] = item

    def append(self,key,item):
        """
        追加一个数据
        :param key:
        :param item:
        :return:
        """
        if key in self.data:
            self.data[key].append(item)
        else:
            self.data[key] = [item]

    def to_file(self):
        if os.path.exists(self.file_dir):
            os.remove(self.file_dir)
        json_data = json.dumps(self.data,indent=4)
        with open(self.file_dir,"w") as json_file:
            json_file.write(json_data)