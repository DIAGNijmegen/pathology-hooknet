import numpy as np
from wholeslidedata.iterators.batchiterator import BatchIterator


class HookNetBatchIterator(BatchIterator):
    def __next__(self):
        x_batch, y_batch, _ = super().__next__()
        x_batch = np.transpose(x_batch, (1, 0, 4, 2, 3))  # inputs first
        y_batch = np.transpose(y_batch, (1, 0, 2, 3))[0]  # inputs first and high-res label only
        return x_batch, y_batch