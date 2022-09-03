import os
import sys

import time

def timeit(func):
    def new_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} s'.format(
            func.__name__, int(elapsed_time)))
        return result
    return new_func
