from collections import defaultdict


class IndependenceFilter(object):
    def __init__(self, d):
        self._filter = d

    @classmethod
    def from_self(cls, id_list):
        d = dict((i, i) for i in id_list)
        return cls(d)

    @classmethod
    def from_common_lookup(cls, id_list, lookup_list):
        zipped = list(zip(id_list, lookup_list))
        d = dict((i, l) for i, l in zipped)
        reverse_d = defaultdict(list)
        for i, l in zipped:
            reverse_d[l].append(i)
        d2 = dict((i, reverse_d[l]) for i, l in d.items())
        return cls(d2)

    def mask(self, target, seq):
        return [False if x in self._filter[target] else True for x in seq]
