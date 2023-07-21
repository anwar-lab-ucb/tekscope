"""
Utilities for saving oscilloscope data to disk.
"""

import struct
import numpy as np

from .waveform import WaveformMetadata, Waveform, Waveforms

# TODO: Currently only works if raw datapoints are 8 bits long.
def write_waveform(file, waveform: Waveform):
    """
    Save a single waveform to an IO stream.
    """
    channel_bytes = waveform.channel.encode("utf-8")
    channel_len = struct.pack("<B", len(channel_bytes))
    channel = struct.pack(f"<{len(channel_bytes)}B", *channel_bytes)
    t_incr = struct.pack("<d", waveform.metadata.t_incr)
    t_zero = struct.pack("<d", waveform.metadata.t_zero)
    v_mult = struct.pack("<d", waveform.metadata.v_mult)
    v_off = struct.pack("<d", waveform.metadata.v_off)
    v_zero = struct.pack("<d", waveform.metadata.v_zero)
    raw_data_len = struct.pack("<Q", len(waveform.raw_data))
    raw_data = struct.pack(f"<{len(waveform.raw_data)}b", *waveform.raw_data)

    file.write(
        channel_len
        + channel
        + t_incr
        + t_zero
        + v_mult
        + v_off
        + v_zero
        + raw_data_len
        + raw_data
    )


def save_waveform(waveform: Waveform, path: str):
    """
    Save a single waveform to the provided path.
    """
    with open(path, "wb") as file:
        write_waveform(file, waveform)


def write_waveforms(file, waveforms: Waveforms):
    """
    Save multiple waveforms to an IO stream.
    """
    for waveform in waveforms.all():
        write_waveform(file, waveform)


def save_waveforms(waveforms: Waveforms, path: str):
    """
    Save multiple waveforms to the provided path.
    """
    with open(path, "wb") as file:
        write_waveforms(file, waveforms)


# TODO: Currently only works if raw datapoints are 8 bits long.
def read_waveform(file) -> Waveform:
    """
    Loads first waveform in file.
    """
    channel_len_packed = file.read(struct.calcsize("<B"))
    if channel_len_packed == b"":
        return None
    channel_len = struct.unpack("<B", channel_len_packed)[0]
    channel_fmt = f"<{channel_len}c"
    channel = b"".join(
        struct.unpack(channel_fmt, file.read(struct.calcsize(channel_fmt)))
    ).decode("utf-8")
    t_incr = struct.unpack("<d", file.read(struct.calcsize("<d")))[0]
    t_zero = struct.unpack("<d", file.read(struct.calcsize("<d")))[0]
    v_mult = struct.unpack("<d", file.read(struct.calcsize("<d")))[0]
    v_off = struct.unpack("<d", file.read(struct.calcsize("<d")))[0]
    v_zero = struct.unpack("<d", file.read(struct.calcsize("<d")))[0]
    raw_data_len = struct.unpack("<Q", file.read(struct.calcsize("<Q")))[0]

    raw_data_fmt = f"<{raw_data_len}b"
    raw_data = np.array(struct.unpack(raw_data_fmt, file.read(struct.calcsize(raw_data_fmt))), dtype=np.int8)

    return Waveform(
        channel, WaveformMetadata(t_incr, t_zero, v_mult, v_off, v_zero), raw_data
    )


def load_waveform(path: str) -> Waveform:
    """
    Load first wavefrom from the provided path
    """
    with open(path, "rb") as file:
        return read_waveform(file)


def read_waveforms(file) -> Waveforms:
    """
    Load waveforms from a file.
    """
    waveforms = []
    waveform = read_waveform(file)
    while waveform is not None:
        waveforms.append(waveform)
        waveform = read_waveform(file)

    return Waveforms(waveforms)


def load_waveforms(path: str) -> Waveforms:
    """
    Load multiple waveforms from from the provided path
    """
    with open(path, "rb") as file:
        return read_waveforms(file)
