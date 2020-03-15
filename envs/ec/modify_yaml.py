import yaml


class ModifyYAML:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

        self.__load()

    def __load(self):
        with open(self.file_path, 'r', encoding="utf-8") as f:
            self.data = yaml.load(f)

    def dump(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.data, f)
