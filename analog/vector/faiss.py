import faiss
import numpy as np

from analog.vector import VectorHandler


class FaissIndexWrapper(VectorHandler):
    def __init__(self, d, index_type="FlatIP", nlist=None):
        self.d = d  # dimension of the vectors
        self.index_type = index_type
        self.nlist = nlist  # only used for the number of clusters for IndexIVFFlat.
        self.is_trained = False
        self.train_needed = False
        self.index = self._create_index()
        self.db = self.index

    # TODO: Configure the best index scheme for our use cases.
    def _create_index(self):
        if self.index_type == "FlatIP":
            return faiss.IndexFlatIP(self.d)
        elif self.index_type == "IVFFlat":
            assert self.nlist is not None, "nlist cannot be None for IVFFlat"
            # TODO: Figure what quantizer may be the most approporaite.
            quantizer = faiss.IndexFlatL2(self.d)  # using L2 as the quantizer.
            self.train_needed = True
            return faiss.IndexIVFFlat(quantizer, self.d, self.nlist, faiss.METRIC_L2)
        else:
            raise ValueError("Unsupported index type")

    # TODO: Figure out the adequate training configurations.
    def train_index(self, training_data):
        assert training_data.shape[1] == self.d, "Training data's dimension mismatch"
        self.index.train(training_data)
        self.is_trained = True

    def add_vectors(self, vectors):
        self._check_valid_index()

        assert vectors.shape[1] == self.d, "Vector dimension mismatch"
        self.index.add(vectors)

    def query_vectors(self, vectors, top_k):
        self._check_valid_index()

        assert vectors.shape[1] == self.d, "Query vector dimension mismatch"
        distances, indices = self.index.search(vectors, top_k)
        return indices, distances

    def write_to_disk(self, file_path):
        self._check_valid_index()
        faiss.write_index(self.index, file_path)

    def read_from_disk(self, file_path):
        self._check_valid_index()
        self.index = faiss.read_index(file_path)

    def _check_valid_index(self):
        if self.index_type == "IVFFlat" and not self.is_trained:
            raise RuntimeError("IndexIVFFlat needs to be trained before adding vectors")

        if self.index is None:
            raise ValueError("Index has not been initialized or is set to None.")
