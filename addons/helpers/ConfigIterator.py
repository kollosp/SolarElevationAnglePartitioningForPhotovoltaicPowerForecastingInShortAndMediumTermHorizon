class ConfigIterator:
    def __init__(self, indexes = None, args={}, verbose=0, skip=0):
        self._args = args
        self._verbose = verbose
        self._indexes = indexes
        if self._indexes is None:
            self._indexes = {}
            for i in args:
                self._indexes[i] = 0

        self._total_iterations = 1
        self._current_iteration = 0
        for i in args:
            self._total_iterations *= len(args[i])

        for _ in range(skip):
            self.increase_indexes()
        self._current_iteration = skip

    def increase_indexes(self):
        ret = {}
        names = [i for i in self._args]
        for i, name in enumerate(names):
            if self._indexes[name] >= len(self._args[name]):
                self._indexes[name] = 0
                if i + 1 < len(names):
                    self._indexes[names[i + 1]] += 1
                else:
                    raise StopIteration()

            ret[name] = self._args[name][self._indexes[name]]

            if i == 0:
                self._indexes[name] += 1
        return ret

    def __iter__(self):
        return self

    def __next__(self):


        ret = self.increase_indexes()

        self._current_iteration += 1
        if self._verbose > 0:
            print(f"ConfigIterator: running iteration {self._current_iteration}/{self._total_iterations}. Current configuration: {ret}")

        cc = {}
        # flat dictionary. Some elements can be linked. Like positon and dataset. Then this kind of parameter shoudl be
        # stored as an object in kwargs
        for j in ret:
            if isinstance(ret[j], dict):
                for k, v in ret[j].items():
                    cc[k] = v
            else:
                cc[j] = ret[j]

        return ret, self._indexes, cc