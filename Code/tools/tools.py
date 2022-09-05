import os
import sys

import time

#new trick
def timeit(func):
    """time measure function"""
    def new_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} s'.format(
            func.__name__, int(elapsed_time)))
        return result
    return new_func

#new functions found, insert, set path: 

#REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))