import os
import sys
from tqdm import tqdm

stdout = sys.stdout

class Logger(object):
    def __init__(self, log_filename):
        self.log = open(log_filename, "a")
        self.count = 0

    def write(self, message):
        tqdm.write(message, file=stdout, end='')
        self.log.write(message)  
        self.count += 1
        if self.count % 5 == 0:
            self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.flush()