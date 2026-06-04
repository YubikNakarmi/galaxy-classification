import os
import glob
import tempfile

class directory():
    def __init__(self, path):
        self.path = path

    def create(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def delete(self):
        if os.path.exists(self.path):
            os.rmdir(self.path)

    def list_files(self, extension=None):
        if extension:
            return glob.glob(os.path.join(self.path, f'*.{extension}'))
        else:
            return glob.glob(os.path.join(self.path, '*'))
        
    def check(self):
        return os.path.exists(self.path)