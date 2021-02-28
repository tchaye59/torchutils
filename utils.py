import random


class RandomBalancedSampler():

    def __init__(self, idx, label, batch_size):
        self.batch_size = batch_size
        indexes = dict([(l, []) for l in set(label)])
        for i, y in zip(idx, label):
            indexes[y].append(i)
        self.indexes = indexes
        self.size = len(idx)
        self.num_iter = 0

    def sample(self):
        labels = list(self.indexes.keys())
        l_size = len(labels)
        while self.num_iter <= len(self):
            lidx = random.randint(0, l_size)
            arr = [random.choice(self.indexes[labels[i % l_size]]) for i in range(lidx, lidx + self.batch_size)]
            random.shuffle(arr)
            assert len(arr) == self.batch_size, self.batch_size
            yield arr
            self.num_iter += 1
        self.num_iter = 0

    def __iter__(self):
        return iter(self.sample())

    def __len__(self):
        return self.size // self.batch_size
