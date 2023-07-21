"""
Tests for saving and restoring data to/from disk.
"""

import os
import io
import numpy as np

from tekscope.waveform import Waveform, Waveforms, WaveformMetadata
from tekscope.io import save_waveforms, load_waveforms, write_waveforms, read_waveforms

from .context import BUILD_DIR


def test_write_and_read_waveforms():
    """
    Test the `write_waveforms` and `read_waveforms` functions.
    """
    file = io.BytesIO()

    metadata1 = WaveformMetadata(1.6e-9, -504e-6, 20e-3, -125, 0)
    raw_data1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    wf1 = Waveform("CH1", metadata1, raw_data1)

    metadata2 = WaveformMetadata(1.6e-9, -504e-6, 80e-3, 0, 0)
    raw_data2 = np.array([8, 7, 6, 5, 4, 3, 2, 1])
    wf2 = Waveform("CH2", metadata2, raw_data2)

    waveforms = Waveforms([wf1, wf2])

    write_waveforms(file, waveforms)

    file.seek(0)

    loaded_waveforms = read_waveforms(file)

    loaded_wf1 = loaded_waveforms.get("CH1")
    loaded_wf2 = loaded_waveforms.get("CH2")

    assert np.all(loaded_wf1.raw_data == raw_data1)
    assert np.isclose(loaded_wf1.metadata.t_incr, 1.6e-9)
    assert np.isclose(loaded_wf1.metadata.t_zero, -504e-6)
    assert np.isclose(loaded_wf1.metadata.v_mult, 20e-3)
    assert np.isclose(loaded_wf1.metadata.v_off, -125)
    assert np.isclose(loaded_wf1.metadata.v_zero, 0)

    assert np.all(loaded_wf2.raw_data == raw_data2)
    assert np.isclose(loaded_wf2.metadata.t_incr, 1.6e-9)
    assert np.isclose(loaded_wf2.metadata.t_zero, -504e-6)
    assert np.isclose(loaded_wf2.metadata.v_mult, 80e-3)
    assert np.isclose(loaded_wf2.metadata.v_off, 0)
    assert np.isclose(loaded_wf2.metadata.v_zero, 0)


def test_save_and_load_waveforms():
    """
    Test the `write_waveforms` and `read_waveforms` functions.
    """
    metadata1 = WaveformMetadata(1.6e-9, -504e-6, 20e-3, -125, 0)
    raw_data1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    wf1 = Waveform("CH1", metadata1, raw_data1)

    metadata2 = WaveformMetadata(1.6e-9, -504e-6, 80e-3, 0, 0)
    raw_data2 = np.array([8, 7, 6, 5, 4, 3, 2, 1])
    wf2 = Waveform("CH2", metadata2, raw_data2)

    waveforms = Waveforms([wf1, wf2])

    save_path = os.path.join(BUILD_DIR, "test_save_and_load_waveforms.tek")
    save_waveforms(waveforms, save_path)

    loaded_waveforms = load_waveforms(save_path)

    loaded_wf1 = loaded_waveforms.get("CH1")
    loaded_wf2 = loaded_waveforms.get("CH2")

    assert np.all(loaded_wf1.raw_data == raw_data1)
    assert np.isclose(loaded_wf1.metadata.t_incr, 1.6e-9)
    assert np.isclose(loaded_wf1.metadata.t_zero, -504e-6)
    assert np.isclose(loaded_wf1.metadata.v_mult, 20e-3)
    assert np.isclose(loaded_wf1.metadata.v_off, -125)
    assert np.isclose(loaded_wf1.metadata.v_zero, 0)

    assert np.all(loaded_wf2.raw_data == raw_data2)
    assert np.isclose(loaded_wf2.metadata.t_incr, 1.6e-9)
    assert np.isclose(loaded_wf2.metadata.t_zero, -504e-6)
    assert np.isclose(loaded_wf2.metadata.v_mult, 80e-3)
    assert np.isclose(loaded_wf2.metadata.v_off, 0)
    assert np.isclose(loaded_wf2.metadata.v_zero, 0)
