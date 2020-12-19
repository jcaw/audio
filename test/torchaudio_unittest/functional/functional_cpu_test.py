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


def _make_spectrogram(items, channels, wave_len, scale=1.):
    # Make some noise, -1 to 1
    wave = (torch.rand(items, channels, wave_len) * 2 - 1) * scale
    spectrogram = torchaudio.transforms.Spectrogram()
    return wave, spectrogram(wave)


class Test_amplitude_to_DB(common_utils.TorchaudioTestCase):
    AMPLITUDE_MULT = 20.
    POWER_MULT = 10.
    AMIN = 1e-10
    REF = 1.0
    # FIXME: So the DB multiplier is 0?
    DB_MULTIPLIER = math.log10(max(AMIN, REF))

    def _ensure_reversible(self, wave, spec, batch):
        # Waveform amplitude -> DB -> amplitude
        multiplier = self.AMPLITUDE_MULT
        power = 0.5

        # FIXME: Why is wave no longer working?
        # db = F.amplitude_to_DB(torch.abs(wave), multiplier, self.AMIN, self.DB_MULTIPLIER,
        #                        top_db=None, batch=batch)
        # x2 = F.DB_to_amplitude(db, self.REF, power)

        # torch.testing.assert_allclose(x2, torch.abs(wave), atol=5e-5, rtol=1e-5)

        # Spectrogram amplitude -> DB -> amplitude
        db = F.amplitude_to_DB(spec, multiplier, self.AMIN, self.DB_MULTIPLIER,
                               top_db=None, batch=batch)
        x2 = F.DB_to_amplitude(db, self.REF, power)

        torch.testing.assert_allclose(x2, spec, atol=5e-5, rtol=1e-5)

        # Waveform power -> DB -> power
        multiplier = self.POWER_MULT
        power = 1.

        # db = F.amplitude_to_DB(wave, multiplier, self.AMIN, self.DB_MULTIPLIER,
        #                        top_db=None, batch=batch)
        # x2 = F.DB_to_amplitude(db, self.REF, power)

        # torch.testing.assert_allclose(x2, torch.abs(wave), atol=5e-5, rtol=1e-5)

        # Spectrogram power -> DB -> power
        db = F.amplitude_to_DB(spec, multiplier, self.AMIN, self.DB_MULTIPLIER,
                               top_db=None, batch=batch)
        x2 = F.DB_to_amplitude(db, self.REF, power)

        torch.testing.assert_allclose(x2, spec, atol=5e-5, rtol=1e-5)

    def test_amplitude_to_DB(self):
        wave, spec = _make_spectrogram(2, 2, 1000, scale=0.5)
        self._ensure_reversible(wave[0], spec[0], batch=False)
        self._ensure_reversible(wave, spec, batch=True)

    def test_top_db(self):
        top_db = 40.

        _, spec = _make_spectrogram(1, 2, 1000, scale=0.5)
        # Make the max value predictable.
        spec[0,0,0] = 200

        decibels = F.amplitude_to_DB(spec[0], self.AMPLITUDE_MULT, self.AMIN,
                                     self.DB_MULTIPLIER, top_db=top_db, batch=False)
        decibels_batch = F.amplitude_to_DB(spec, self.AMPLITUDE_MULT, self.AMIN,
                                           self.DB_MULTIPLIER, top_db=top_db, batch=True)

        # The actual db floor will be just below 6.0206 - use 6.0205 to deal
        # with rounding error.
        above_top = decibels >= 6.0205
        assert above_top.all(), decibels
        above_top_batch = decibels_batch >= 6.0205
        assert above_top_batch.all(), decibels_batch

    def test_batched(self):
        top_db = 40.

        # Make a batch of noise
        _, spec = _make_spectrogram(2, 2, 1000)
        # Make the second item blow out the first
        spec[0] *= 0.5

        # Ensure the clamp applies per-item, not at the batch level.
        batchwise_dbs = F.amplitude_to_DB(spec, self.AMPLITUDE_MULT, self.AMIN,
                                          self.DB_MULTIPLIER, top_db=top_db, batch=True)
        itemwise_dbs = [
            F.amplitude_to_DB(item, self.AMPLITUDE_MULT, self.AMIN,
                              self.DB_MULTIPLIER, top_db=top_db, batch=False)
            for item in spec
        ]

        torch.testing.assert_allclose(
            batchwise_dbs, torch.stack(itemwise_dbs), atol=5e-5, rtol=1e-5
        )

    def test_per_spectrogram(self):
        channels = 2
        top_db = 80.

        _, spec = _make_spectrogram(1, channels, 1000)
        # Make the second channel blow out the first
        spec[:, 0] *= 0.5

        specwise_dbs = F.amplitude_to_DB(spec, self.AMPLITUDE_MULT, self.AMIN,
                                         self.DB_MULTIPLIER, top_db=top_db)
        channelwise_dbs = [
            F.amplitude_to_DB(spec[:, i], self.AMPLITUDE_MULT, self.AMIN,
                              self.DB_MULTIPLIER, top_db=top_db, batch=False)
            for i in range(channels)
        ]

        # Just check channelwise gives a different answer.
        difference = (specwise_dbs - torch.stack(channelwise_dbs)).abs()
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
