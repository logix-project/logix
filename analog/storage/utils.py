import msgpack
import msgpack_numpy as mn
import lz4.frame

from einops import rearrange


def msgpack_serialize(obj):
    return lz4.frame.compress(msgpack.packb(obj, default=mn.encode))


def msgpack_deserialize(obj):
    return msgpack.unpackb(lz4.frame.decompress(obj), object_hook=mn.decode)
