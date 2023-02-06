import configparser

class Config:
    def __init__(self, fname='config.txt'):
        self.fname = fname
        self.config = configparser.ConfigParser()

    def read_config(self):
        self.config.read(self.fname)
        return self.get_config()
        
    def get_config(self):
        return self.config