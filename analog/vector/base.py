from abc import ABC, abstractmethod


class VectorHandler(ABC):
    def __init__(self, d):
        self.d = d  # Dimension of vectors.
        self.db = None  # Is index for FIASS, collection for chroma DB.
        self.length = 0  # Number of vectors in the index.

        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """
        Abstract method to handle initialization. This is always called when class is constructed.
        """
        pass

    @abstractmethod
    def add_vectors(self, vectors):
        """
        Args:
           vectors (np.array): (,self.d) vectors to add.
        """
        pass

    @abstractmethod
    def query_vectors(self, vectors, top_k):
        """
        search_vector searches the given vectors from the index which returns at most top_k vectors per query.

        Args:
            vectors (np.array): (,self.d) sized vectors to search from the DB.
            top_k (int): number of extracted vectors per query.

        Returns:
            tuple of np.array, indices and distances of the result.
            indices: indices of query result.
            distances: distances of input query to the output indices.
        """
        pass

    @abstractmethod
    def write_to_disk(self, file_path):
        """
        write_to_disk writes the index from memory to path.

        Args:
            file_path (str): path to save the index.
        """
        pass

    @abstractmethod
    def read_from_disk(self, file_path):
        """
        read_from_disk load the index to memory from path.

        Args:
            file_path (str): path to load from.
        """
        pass

    def get_db(self):
        """
        get_db gets database entity such as index or collection.

        Returns(object):
            the database object.
        """
        return self.db

    def get_size(self):
        """
        get_size gets length of the vectors in the database.
        Returns:
            length of the vectors of the index.
        """
        return self.length
