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


def save_to_mmap(data_buffer, chunk_index, log_dir, dtype="uint8"):
    mmap_filename = str(os.path.join(log_dir, f"log_chunk_{chunk_index}.mmap"))
    schema_filename = str(os.path.join(log_dir, f"metadata_chunk_{chunk_index}.json"))

    total_size = sum(arr.nbytes for _, d in data_buffer for _, arr in extract_arrays(d))
    mmap = np.memmap(mmap_filename, dtype=dtype, mode="w+", shape=(total_size,))

    schema = []
    offset = 0

    for data_id, nested_dict in data_buffer:
        for path, arr in extract_arrays(nested_dict):
            mmap[offset : offset + arr.nbytes] = arr.ravel().view(dtype)
            schema.append(
                {
                    "data_id": data_id,
                    "path": path,
                    "offset": offset,
                    "shape": arr.shape,
                    "dtype": str(arr.dtype),
                }
            )
            offset += arr.nbytes

    mmap.flush()
    del mmap  # Release the memmap object

    with open(schema_filename, "w") as f:
        json.dump(schema, f, indent=2)
