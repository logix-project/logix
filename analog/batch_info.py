from analog.utils import nested_dict


class BatchInfo:
    def __init__(self):
        self._data_id = None
        self._mask = None
        self._log = nested_dict()

    def clear(self):
        self._data_id = None
        self._mask = None
        self._log.clear()

    @property
    def data_id(self):
        return self._data_id

    @data_id.setter
    def data_id(self, data_id):
        self._data_id = data_id

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    @property
    def log(self):
        return self.log
