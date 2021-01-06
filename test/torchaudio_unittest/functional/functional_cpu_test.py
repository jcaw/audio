import math
import unittest

import torch
import torchaudio
import torchaudio.functional as F
from parameterized import parameterized
import pytest

from torchaudio_unittest import common_utils
from .functional_impl import Lfilter


class TestLFilterFloat32(Lfilter, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestLFilterFloat64(Lfilter, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


class TestCreateFBMatrix(common_utils.TorchaudioTestCase):
    def test_no_warning_high_n_freq(self):
        with pytest.warns(None) as w:
            F.create_fb_matrix(288, 0, 8000, 128, 16000)
        assert len(w) == 0

    def test_no_warning_low_n_mels(self):
        with pytest.warns(None) as w:
            F.create_fb_matrix(201, 0, 8000, 89, 16000)
        assert len(w) == 0

    def test_warning(self):
        with pytest.warns(None) as w:
            F.create_fb_matrix(201, 0, 8000, 128, 16000)
        assert len(w) == 1


class TestComputeDeltas(common_utils.TorchaudioTestCase):
    """Test suite for correctness of compute_deltas"""
    def test_one_channel(self):
        specgram = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5]]])
        computed = F.compute_deltas(specgram, win_length=3)
        torch.testing.assert_allclose(computed, expected)

    def test_two_channels(self):
        specgram = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                  [1.0, 2.0, 3.0, 4.0]]])
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5],
                                  [0.5, 1.0, 1.0, 0.5]]])
        computed = F.compute_deltas(specgram, win_length=3)
        torch.testing.assert_allclose(computed, expected)


class TestDetectPitchFrequency(common_utils.TorchaudioTestCase):
    @parameterized.expand([(100,), (440,)])
    def test_pitch(self, frequency):
        sample_rate = 44100
        test_sine_waveform = common_utils.get_sinusoid(
            frequency=frequency, sample_rate=sample_rate, duration=5,
        )

        freq = torchaudio.functional.detect_pitch_frequency(test_sine_waveform, sample_rate)

        threshold = 1
        s = ((freq - frequency).abs() > threshold).sum()
        self.assertFalse(s)


class Testamplitude_to_DB(common_utils.TorchaudioTestCase):
    AMPLITUDE_MULT = 20.
    POWER_MULT = 10.
    AMIN = 1e-10
    REF = 1.0
    # FIXME: So the DB multiplier is 0?
    DB_MULT = math.log10(max(AMIN, REF))

    def _ensure_reversible(self, spec):
        """Check `amplitude_to_db` returns the original when reversed."""
        # Spectrogram amplitude -> DB -> amplitude
        db = F.amplitude_to_DB(spec, self.AMPLITUDE_MULT, self.AMIN, self.DB_MULT, top_db=None)
        x2 = F.DB_to_amplitude(db, self.REF, 0.5)

        torch.testing.assert_allclose(x2, spec, atol=5e-5, rtol=1e-5)

        # Spectrogram power -> DB -> power
        db = F.amplitude_to_DB(spec, self.POWER_MULT, self.AMIN, self.DB_MULT, top_db=None)
        x2 = F.DB_to_amplitude(db, self.REF, 1.)

        torch.testing.assert_allclose(x2, spec, atol=5e-5, rtol=1e-5)

    def test_amplitude_to_DB_batch(self):
        spec = torch.rand([2, 2, 100, 100]) * 200
        self._ensure_reversible(spec)

    def test_amplitude_to_DB_channels(self):
        spec = torch.rand([2, 100, 100]) * 200
        self._ensure_reversible(spec)

    def test_amplitude_to_DB_item(self):
        spec = torch.rand([100, 100]) * 200
        self._ensure_reversible(spec)

    def test_top_db(self):
        top_db = 40.

        spec = torch.rand([1, 2, 100, 100]) * 200
        # Make the max value (and thus DB cutoff) predictable.
        spec[0,0,0] = 200

        decibels = F.amplitude_to_DB(spec[0], self.AMPLITUDE_MULT, self.AMIN,
                                     self.DB_MULT, top_db=top_db)
        # The actual db floor will be just below 6.0206 - use 6.0205 to deal
        # with rounding error.
        above_top = decibels >= 6.0205
        assert above_top.all(), decibels

        # And check it works with batch dimension
        decibels_batch = F.amplitude_to_DB(spec, self.AMPLITUDE_MULT, self.AMIN,
                                           self.DB_MULT, top_db=top_db)
        above_top_batch = decibels_batch >= 6.0205
        assert above_top_batch.all(), decibels_batch

        # And check it works with shape (freq, time) only
        decibels_batch = F.amplitude_to_DB(spec[0][0], self.AMPLITUDE_MULT, self.AMIN,
                                           self.DB_MULT, top_db=top_db)
        above_top_batch = decibels_batch >= 6.0205
        assert above_top_batch.all(), decibels_batch

    def test_batched(self):
        top_db = 40.

        # Make a batch of noise
        spec = torch.rand([2, 2, 100, 100]) * 200
        # Make the second item blow out the first
        spec[0] *= 0.5
        # Predictability
        spec[1, 0, 0, 0] = 200

        # Ensure the clamp applies per-item, not at the batch level.
        batchwise_dbs = F.amplitude_to_DB(spec, self.AMPLITUDE_MULT, self.AMIN,
                                          self.DB_MULT, top_db=top_db)
        itemwise_dbs = torch.stack([
            F.amplitude_to_DB(item, self.AMPLITUDE_MULT, self.AMIN,
                              self.DB_MULT, top_db=top_db)
            for item in spec
        ])

        torch.testing.assert_allclose(batchwise_dbs, itemwise_dbs, atol=5e-5, rtol=1e-5)

    def test_per_spectrogram(self):
        channels = 2
        top_db = 40.

        spec = torch.rand([1, channels, 100, 100]) * 200
        # Make the second channel blow out the first
        spec[:, 0] *= 0.5
        # Predictability
        spec[0, 1, 0, 0] = 200

        # Ensure the clamp applies per-item, not per-channel.
        specwise_dbs = F.amplitude_to_DB(spec, self.AMPLITUDE_MULT, self.AMIN,
                                         self.DB_MULT, top_db=top_db)
        channelwise_dbs = torch.stack([
            F.amplitude_to_DB(spec[:, i], self.AMPLITUDE_MULT, self.AMIN,
                              self.DB_MULT, top_db=top_db)
            for i in range(channels)
        ])

        # Just check channelwise gives a different answer.
        difference = (specwise_dbs - channelwise_dbs).abs()
        assert (difference >= 1e-5).any()


@pytest.mark.parametrize('complex_tensor', [
    torch.randn(1, 2, 1025, 400, 2),
    torch.randn(1025, 400, 2)
])
@pytest.mark.parametrize('power', [1, 2, 0.7])
def test_complex_norm(complex_tensor, power):
    expected_norm_tensor = complex_tensor.pow(2).sum(-1).pow(power / 2)
    norm_tensor = F.complex_norm(complex_tensor, power)

    torch.testing.assert_allclose(norm_tensor, expected_norm_tensor, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('specgram', [
    torch.randn(2, 1025, 400),
    torch.randn(1, 201, 100)
])
@pytest.mark.parametrize('mask_param', [100])
@pytest.mark.parametrize('mask_value', [0., 30.])
@pytest.mark.parametrize('axis', [1, 2])
def test_mask_along_axis(specgram, mask_param, mask_value, axis):

    mask_specgram = F.mask_along_axis(specgram, mask_param, mask_value, axis)

    other_axis = 1 if axis == 2 else 2

    masked_columns = (mask_specgram == mask_value).sum(other_axis)
    num_masked_columns = (masked_columns == mask_specgram.size(other_axis)).sum()
    num_masked_columns //= mask_specgram.size(0)

    assert mask_specgram.size() == specgram.size()
    assert num_masked_columns < mask_param


@pytest.mark.parametrize('mask_param', [100])
@pytest.mark.parametrize('mask_value', [0., 30.])
@pytest.mark.parametrize('axis', [2, 3])
def test_mask_along_axis_iid(mask_param, mask_value, axis):
    torch.random.manual_seed(42)
    specgrams = torch.randn(4, 2, 1025, 400)

    mask_specgrams = F.mask_along_axis_iid(specgrams, mask_param, mask_value, axis)

    other_axis = 2 if axis == 3 else 3

    masked_columns = (mask_specgrams == mask_value).sum(other_axis)
    num_masked_columns = (masked_columns == mask_specgrams.size(other_axis)).sum(-1)

    assert mask_specgrams.size() == specgrams.size()
    assert (num_masked_columns < mask_param).sum() == num_masked_columns.numel()
