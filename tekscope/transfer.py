"""
Wrapper functions for transferring data to/from the oscilloscope.
"""

import socket
from .raw import (
    send_command,
    query_ascii,
    query_binary,
    data_source_cmd,
    data_start_cmd,
    data_stop_cmd,
    data_width_cmd,
    data_encdg_cmd,
    curve_cmd,
    wfmoutpre_cmd,
    DataEncdg,
    AnalogSource,
    DigitalSource,
)
from .parse import parse_ribinary_seq, parse_wfmoutpre
from .horizontal import record_length
from .waveform import WaveformMetadata, Waveform, Waveforms


def set_data_source(soc: socket.socket, source: str):
    """
    Specifies source waveform to be transferred from the oscilloscope.
    """
    send_command(soc, data_source_cmd(source))


def set_data_start(soc: socket.socket, start: int):
    """
    Specifies starting data point for waveform transfer.
    """
    send_command(soc, data_start_cmd(start))


def set_data_stop(soc: socket.socket, stop: int):
    """
    Specifies final data point for waveform transfer.
    """
    send_command(soc, data_stop_cmd(stop))


def set_data_width(soc: socket.socket, width: int):
    """
    Specifies data width for waveform transfer.
    """
    send_command(soc, data_width_cmd(width))


def set_data_encdg(soc: socket.socket, encdg: str):
    """
    Specifies data encoding format for waveform transfer.
    """
    send_command(soc, data_encdg_cmd(encdg))


def get_curve(soc: socket.socket) -> bytes:
    """
    Retrieves a curve from the oscilloscope following existing data setting.
    """
    send_command(soc, curve_cmd())
    return query_binary(soc)


def get_waveform_metadata(soc: socket.socket) -> WaveformMetadata:
    """
    Retrieves curve parameters from oscilloscope following existing data setting.
    """
    send_command(soc, wfmoutpre_cmd())
    return parse_wfmoutpre(query_ascii(soc))


def retrieve_waveform_with_default_settings(
    soc: socket.socket, source: str
) -> Waveform:
    """
    Helper function that retrieves waveform assuming that correct settings have been applied.
    """
    set_data_source(soc, source)
    metadata = get_waveform_metadata(soc)
    if metadata is None:
        return None
    raw_data = parse_ribinary_seq(get_curve(soc))
    return Waveform(source, metadata, raw_data)


def set_default_waveform_settings(soc: socket.socket):
    """
    Helper function for setting up correct settings for retrieving waveforms.
    """
    samples = record_length(soc)
    set_data_start(soc, 1)
    set_data_stop(soc, samples)
    set_data_width(soc, 1)
    set_data_encdg(soc, DataEncdg.BINARY)


def retrieve_waveform(soc: socket.socket, source: str) -> Waveform:
    """
    Retrieves a waveform from the oscilloscope as a `Waveform` object.
    """
    assert AnalogSource.is_valid(source) or DigitalSource.is_valid(source)
    set_default_waveform_settings(soc)
    return retrieve_waveform_with_default_settings(soc, source)


def retrieve_waveform_parameters(soc: socket.socket, source: str) -> WaveformMetadata:
    """
    Retrieves a waveform's parameters from the oscilloscope as a `WaveformMetadata` object.
    """
    set_default_waveform_settings(soc)
    set_data_source(soc, source)
    return get_waveform_metadata(soc)


def retrieve_all_waveforms(soc: socket.socket) -> Waveforms:
    """
    Retrieves all analog and digital waveforms from the oscilloscope as a `Waveforms` object.
    """
    set_default_waveform_settings(soc)

    waveforms = []
    for source in AnalogSource.SOURCES:
        waveform = retrieve_waveform_with_default_settings(soc, source)
        if waveform is not None:
            waveforms.append(waveform)
    for source in DigitalSource.SOURCES:
        waveform = retrieve_waveform_with_default_settings(soc, source)
        if waveform is not None:
            waveforms.append(waveform)

    return Waveforms(waveforms)
