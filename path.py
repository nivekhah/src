"""
通过当前模块 path.py 可以获取 src 目录下各个目录的绝对路径
"""
import os


class Path:

    @staticmethod
    def get_src_path():
        return os.path.dirname(__file__)

    @staticmethod
    def get_config_path():
        return os.path.join(Path.get_src_path(), "config")

    @staticmethod
    def get_envs_config_path():
        return os.path.join(Path.get_config_path(), "envs")

    @staticmethod
    def get_algs_config_path():
        return os.path.join(Path.get_config_path(), "algs")

    @staticmethod
    def get_envs_path():
        return os.path.join(Path.get_src_path(), "envs")

    @staticmethod
    def get_ec_path():
        return os.path.join(Path.get_envs_path(), "ec")
