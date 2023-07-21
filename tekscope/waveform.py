"""
Types and utilities for storing and interacting with waveforms in memory.
"""

import numpy as np

# pylint: disable-next=too-few-public-methods
class WaveformMetadata:
    """
    Class for storing waveform metadata in memory.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self, t_incr: float, t_zero: float, v_mult: float, v_off: float, v_zero: float
    ):
        """
        Initializes a `WaveformMetadata` object.

        `t_incr`: The time increment between datapoints.
        `t_zero`: The time offset from the trigger of the first datapoint.
        `v_mult`: The multiplicative factor between digitzing levels and volts.
        `v_off`: The offset in digitizing levels.
        `v_zero`: The voltage offset corresponding to quantized 0.
        """

        self.t_incr = t_incr
        self.t_zero = t_zero
        self.v_mult = v_mult
        self.v_off = v_off
        self.v_zero = v_zero


class Waveform:
    """
    Class for storing a waveform in memory.
    """

    def __init__(self, channel: str, metadata: WaveformMetadata, raw_data):
        """
        Initializes a `Waveform` object.

        `channel`: The channel (e.g. "CH1") associated with the data.
        `raw_data`: Raw digitized values from the oscilloscope as a numpy array.
        `metadata`: Waveform metadata describing the relationship between
            digitizing levels and actual time/voltage values.
        """
        self.channel = channel
        self.metadata = metadata
        self.raw_data = np.array(raw_data)

    def time(self):
        """
        Returns a numpy array of timestamps corresponding to each datapoint.
        """
        return self.metadata.t_zero + self.metadata.t_incr * np.arange(len(self.raw_data))

    def voltage(self):
        """
        Returns a numpy array of voltage values corresponding to each datapoint.
        """
        metadata = self.metadata
        return metadata.v_mult * (self.raw_data - metadata.v_off) + metadata.v_zero


class Waveforms:
    """
    Class for storing multiple waveforms in memory.
    """

    def __init__(self, waveforms: [Waveform]):
        """
        Initializes a `Waveforms` object.

        `waveforms`: The waveforms contained in this object.
        """
        self.waveform_dict = {}
        for waveform in waveforms:
            self.waveform_dict[waveform.channel] = waveform

    def get(self, channel: str) -> Waveform:
        """
        Returns a `Waveform` object corresponding to the given channel.

        Returns `None` if no such object exists.
        """
        return self.waveform_dict.get(channel)

    def all(self) -> [Waveform]:
        """
        Returns a list of all waveforms stored in this object.
        """
        return list(self.waveform_dict.values())
