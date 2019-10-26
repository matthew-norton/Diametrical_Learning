
import torch
import numpy as np

from numpy.random import  PCG64 ,Generator
import multiprocessing

import concurrent.futures

class MultithreadedRNG(object):
    def __init__(self, size, seed=None, threads=None):
        rg = PCG64(seed)
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        self._random_generators = [rg]
        last_rg = rg
        for _ in range(0, threads-1):
            new_rg = last_rg.jumped()
            self._random_generators.append(new_rg)
            last_rg = new_rg
        self.size = size
        self.n = np.prod(size)
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.values = np.empty(self.n , dtype='float32')
        self.step = np.ceil(self.n / threads).astype(np.int)

    def fill(self):
        def _fill(random_state, out, first, last):
            Generator(random_state).standard_normal(out=out[first:last] , dtype = 'f')
        futures = {}
        for i in range(self.threads):
            args = (_fill,
                    self._random_generators[i],
                    self.values,
                    i * self.step,
                    (i + 1) * self.step)
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)

        return torch.as_tensor(self.values.reshape(self.size) )
    def __del__(self):
        self.executor.shutdown(False)
