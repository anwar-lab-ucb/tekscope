"""
API for interfacing with a Tektronix oscilloscope.
"""

import socket
from tekscope import raw
from tekscope import parse
from tekscope import transfer
from tekscope import io


class Oscilloscope:
    """
    Class with high-level utilties for connecting to and remotely
    controlling a Tektronix oscilloscope.
    """

    def __init__(self, host="169.254.8.194", port=4000):
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.connect((host, port))
        raw.send_command(self.soc, raw.header_cmd(False))

    def send_raw_command(self, command):
        """
        Sends a raw command to the oscilloscope.
        """
        raw.send_command(self.soc, command)

    def start_acquire(self):
        """
        Starts an acquisition.
        """
        raw.send_command(self.soc, raw.acquire_state_cmd(raw.AcquireState.RUN))

    def stop_acquire(self):
        """
        Stops an acquisition.
        """
        raw.send_command(self.soc, raw.acquire_state_cmd(raw.AcquireState.STOP))

    def acquire_analog_sequence(self, num_acq: int, source: str) -> [int]:
        """
        Acquires an analog sequence of the given length and parses it as a Python list.
        """
        raw.send_command(self.soc, raw.select_cmd(source, True))
        raw.send_command(self.soc, raw.acquire_numsequence_cmd(num_acq))
        raw.send_command(
            self.soc, raw.acquire_stopafter_cmd(raw.AcquireStopAfter.SEQUENCE)
        )
        raw.send_command(self.soc, raw.acquire_state_cmd(raw.AcquireState.RUN))
        raw.send_command(self.soc, raw.data_source_cmd(source))
        raw.send_command(self.soc, raw.data_start_cmd(1))
        raw.send_command(self.soc, raw.data_stop_cmd(num_acq))
        raw.send_command(self.soc, raw.data_width_cmd(1))
        raw.send_command(self.soc, raw.data_encdg_cmd(raw.DataEncdg.BINARY))
        raw.send_command(self.soc, "*OPC?")
        raw.query_ascii(self.soc)
        raw.send_command(self.soc, raw.curve_cmd())
        return parse.parse_ribinary_seq(raw.query_binary(self.soc))

    def retrieve_waveform(self, source: str):
        """
        Retrieves a waveform from the oscilloscope.
        """
        return transfer.retrieve_waveform(self.soc, source)

    def retrieve_waveform_parameters(self, source: str):
        """
        Retrieves a waveform's parameters from the oscilloscope.
        """
        return transfer.retrieve_waveform_parameters(self.soc, source)

    def retrieve_all_waveforms(self):
        """
        Retrieves all waveforms from the oscilloscope.
        """
        return transfer.retrieve_all_waveforms(self.soc)
