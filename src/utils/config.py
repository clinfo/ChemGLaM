import json


class Config:
    def __init__(self, json_file):
        self.__dict__ = json.load(open(json_file))
