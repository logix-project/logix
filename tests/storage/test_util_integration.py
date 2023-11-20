import glob
import unittest
import numpy as np
import os
import shutil
from analog.storage.utils import MemoryMapHandler


def generate_random_arrays(size=10):
    """Generate a list of random arrays of various numeric types."""
    arrays = []
    for dt in [np.float32, np.float64, np.double, np.longdouble]:
        arrays.append(np.random.uniform(0, 1024, size).astype(dt))
    return arrays


def generate_static_arrays():
    """Generate a list of random arrays of various numeric types."""
    arrays = []
    for dt in [np.float32, np.float64, np.double, np.longdouble]:
        arrays.append(
            np.array(
                [0, 1 / 2, 1 / 3, 1 / 9, 1 / 13, np.pi, np.e, np.euler_gamma], dtype=dt
            )
        )
    return arrays


def cleanup(file_path, prefix):
    pattern = os.path.join(file_path, prefix + "*")
    files_to_remove = glob.glob(pattern)

    for file_path in files_to_remove:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error: {file_path} : {e.strerror}")


class TestMemoryMapHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "tests/test_mmap"
        os.makedirs(cls.test_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_write_and_read(self):
        handler = MemoryMapHandler(self.test_dir)

        data_buffer = [
            (i, {"dummy_data": arr}) for i, arr in enumerate(generate_random_arrays())
        ]
        filename = "test_data"

        handler.write(data_buffer, filename)

        mmap, metadata = handler.read(filename)
        for item in metadata:
            offset = item["offset"]
            size = item["size"]
            shape = tuple(item["shape"])
            dtype = np.dtype(item["dtype"])
            expected_data = data_buffer[item["data_id"]][1]["dummy_data"]
            read_data = np.frombuffer(
                mmap, dtype=dtype, count=size // dtype.itemsize, offset=offset
            ).reshape(shape)
            # Test if expected value and read value equals
            self.assertTrue(np.array_equal(read_data, expected_data), "Data mismatch")

    def test_read(self):
        expected_files_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_mmap_data"
        )
        handler = MemoryMapHandler(expected_files_path)

        filename = "test_data"
        data_buffer = [
            (i, {"dummy_data": arr}) for i, arr in enumerate(generate_static_arrays())
        ]

        handler.write(data_buffer, filename)
        mmap, metadata = handler.read(filename)

        expected_mmap, _ = handler.read(
            "expected_data.mmap"
        )  # Using same metadata file.
        for item in metadata:
            offset = item["offset"]
            size = item["size"]
            shape = tuple(item["shape"])
            dtype = np.dtype(item["dtype"])
            test_data = np.frombuffer(
                mmap, dtype=dtype, count=size // dtype.itemsize, offset=offset
            ).reshape(shape)
            expected_data = np.frombuffer(
                mmap, dtype=dtype, count=size // dtype.itemsize, offset=offset
            ).reshape(shape)
            self.assertTrue(np.allclose(test_data, expected_data), "Data mismatch")
        cleanup(expected_files_path, filename)
