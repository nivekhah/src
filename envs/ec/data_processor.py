import time
import os
import json
class DataSaver:
    dir = os.path.join(os.getcwd(),"src","envs","ec","data")

    def __init__(self,func_name):
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = func_name+"_"+now
        self.file_dir = os.path.join(self.dir,self.filename)
        self.data = {}


    def add(self,key,item):
        #添加数据
        self.data[key] = item

    def to_file(self):
        if os.path.exists(self.file_dir):
            os.remove(self.file_dir)
        json_data = json.dumps(self.data,indent=4)
        with open(self.file_dir,"w") as json_file:
            json_file.write(json_data)