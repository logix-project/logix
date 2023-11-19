import os

import json
import numpy as np

import msgpack
import msgpack_numpy as mn
import lz4.frame

from einops import rearrange


def msgpack_serialize(obj):
    return lz4.frame.compress(msgpack.packb(obj, default=mn.encode))


def msgpack_deserialize(obj):
    return msgpack.unpackb(lz4.frame.decompress(obj), object_hook=mn.decode)


def extract_arrays(obj, base_path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from extract_arrays(v, base_path + (k,))
    elif isinstance(obj, np.ndarray):
        yield base_path, obj


class MemoryMapHandler:
    def __init__(self, log_dir, mmap_dtype='uint8'):
        """
        Args:
            save_path (str): The directory of the path to write and read the binaries and the metadata.
            mmap_dtype: The data type that will be used to save the binary into the memory map.
        """
        self.save_path = log_dir
        self.mmap_dtype = mmap_dtype

    def get_path(self):
        return self.save_path

    def write(self, data_buffer, filename):
        file_root, file_ext = os.path.splitext(filename)
        if file_ext == "":
            filename += ".mmap"

        mmap_filename = os.path.join(self.save_path, filename)
        metadata_filename = os.path.join(self.save_path, file_root + "_metadata.json")

        total_size = sum(arr.nbytes for _, d in data_buffer for _, arr in extract_arrays(d))
        mmap = np.memmap(mmap_filename, dtype=self.mmap_dtype, mode="w+", shape=(total_size,))

        metadata = []
        offset = 0

        for data_id, nested_dict in data_buffer:
            for path, arr in extract_arrays(nested_dict):
                bytes = arr.nbytes
                mmap[offset: offset + bytes] = arr.ravel().view(self.mmap_dtype)
                metadata.append(
                    {
                        "data_id": data_id,
                        "size": bytes,
                        "path": path,
                        "offset": offset,
                        "shape": arr.shape,
                        "dtype": str(arr.dtype),
                    }
                )
                offset += arr.nbytes

        mmap.flush()
        del mmap  # Release the memmap object

        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)

    def read(self, filename):
        """
        read reads the file by chunk index, it will return the data_buffer with metadata.
        Arg:
            filename (str): filename for the path to mmap.
        Returns:
            mmap (np.mmap): memory mapped buffer read from filename.
            metadata (json):
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext == "":
            filename += ".mmap"

        mmap = np.memmap(os.path.join(self.save_path, filename), dtype=self.mmap_dtype, mode="r")
        metadata = self.read_metafile(file_root + "_metadata.json")
        return mmap, metadata

    def read_metafile(self, meta_filename):
        file_root, file_ext = os.path.splitext(meta_filename)
        if file_ext == "":
            meta_filename += ".json"
        f = open(os.path.join(self.save_path, meta_filename), "r")
        metadata = json.load(f)  # This throws error when file does not exist.
        f.close()
        return metadata
