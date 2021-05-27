"""Tests for UHD object.

This module contains unit-tests for the UHD object and its methods.

TODO:
  * write_register
  * read_register
  * enumerate_registers
  * set_user_register
"""

import uhd
import unittest
from itertools import combinations, product
import numpy as np
from math import isclose
from time import sleep


def not_supported(function):
    """Catch "not supported" on device exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except uhd.UhdError as e:
            if 'not supported on this device' not in str(e):
                raise
    return wrapper


def is_range(obj):
    """Helper function to test if object is valid "range"."""
    keys = ['start', 'step', 'stop']
    return isinstance(obj, dict) and all(k in obj for k in keys) and \
           all(isinstance(obj[k], float) for k in keys)


class UhdTestCase(unittest.TestCase):

    def setUp(self):
        """Create UHD object. Do some global setup."""
        self.dut = uhd.Uhd()
        # Setup TX/RX frequencies: some functions need this setup to pass
        for chan in range(self.dut.get_rx_num_channels()):
            self.dut.set_rx_freq(self.dut.get_rx_freq_range(chan)['start'],
                                 chan)
        for chan in range(self.dut.get_tx_num_channels()):
            self.dut.set_tx_freq(self.dut.get_tx_freq_range(chan)['start'],
                                 chan)

    def tearDown(self):
        """Delete UHD object."""
        del self.dut

    def helper_range(self, f_set, f_get, f_range, steps=None):
        """ Given a set, get and range function: test a discrete
        range interface."""
        r = f_range()
        if is_range(r):
            self.assertTrue(is_range(r))
            values = np.arange(r['start'], r['stop'], r['step']) if steps is \
                     None else np.linspace(r['start'], r['stop'], steps)
            values = [float(v) for v in values]
        elif isinstance(r, list):
            values = r
        else:
            raise ValueError('Invalid range-type.')
        for value in values:
            f_set(value)
            ret = f_get()
            if is_range(r) and not isclose(r['step'], 0.):
                self.assertLessEqual(abs(ret - value), r['step'])
            elif isinstance(r, list):
                self.assertEqual(ret, value)
            self.assertIsInstance(ret, type(values[0]))

    def test_versions(self):
        for attr in ('__version__', 'UHD_VERSION_ABI', 'UHD_VERSION_LONG'):
            value = getattr(uhd, attr)
            print('Got {} = \'{}\''.format(attr, value))
            self.assertIsInstance(value, str)
            self.assertTrue(len(value) > 0)
        uhd_ver = uhd.UHD_VERSION
        print('Got UHD_VERSION = {}'.format(uhd_ver))
        self.assertIsInstance(uhd_ver, int)
        self.assertTrue(uhd_ver > 0)

    def test_clock_sources(self):
        """Test set/get clock sources methods."""
        sources = self.dut.get_clock_sources(0)
        self.assertIsInstance(sources, list)
        self.assertIn('internal', sources)
        self.dut.set_clock_source('internal')
        self.assertEqual(self.dut.get_clock_source(0), 'internal')

    @not_supported
    def test_clock_source_out(self):
        """Test clock-source output enable method."""
        for out in [True, False]:
            self.dut.set_clock_source_out(out)

    def test_time_sources(self):
        """Test set/get time sources methods."""
        sources = self.dut.get_time_sources(0)
        self.assertIsInstance(sources, list)
        self.assertIn('internal', sources)
        self.dut.set_time_source('internal')
        self.assertEqual(self.dut.get_time_source(0), 'internal')

    @not_supported
    def test_time_source_out(self):
        """Test time-source output enable method."""
        for out in [True, False]:
            self.dut.set_time_source_out(out)

    def test_master_clock_rate(self):
        """Test methods to set/get master-clock-rate."""
        rate = self.dut.get_master_clock_rate()
        self.assertIsInstance(rate, float)
        self.dut.set_master_clock_rate(rate)

    def test_normalized_rx_gain(self):
        """Test methods to set/get normalized RX gain."""
        for chan in range(self.dut.get_rx_num_channels()):
            self.helper_range(lambda x: self.dut.set_normalized_rx_gain(x, chan),
                              lambda: self.dut.get_normalized_rx_gain(chan),
                              lambda: {'start': 0., 'step': 0., 'stop': 1.},
                              steps=10)

    def test_normalized_tx_gain(self):
        """Test methods to set/get normalized TX gain."""
        for chan in range(self.dut.get_rx_num_channels()):
            self.helper_range(lambda x: self.dut.set_normalized_tx_gain(x, chan),
                              lambda: self.dut.get_normalized_tx_gain(chan),
                              lambda: {'start': 0., 'step': 0., 'stop': 1.},
                              steps=10)

    def test_rx_freq(self):
        """Test methods to get/set/enumerate RX frequency."""
        for chan in range(self.dut.get_rx_num_channels()):
            self.helper_range(lambda x: self.dut.set_rx_freq(x, chan),
                              lambda: self.dut.get_rx_freq(chan),
                              lambda: self.dut.get_rx_freq_range(chan),
                              steps=10)
            r = self.dut.get_rx_freq_range(chan)
            target_freq = round((r['start'] + r['stop']) / 2. / 1.e9) * 1.e9
            for policy in ['none', 'auto', 'manual']:
                tune_result = self.dut.set_rx_freq({'target_freq': target_freq,
                    'lo_off': 0., 'rf_freq_policy': 'manual', 'rf_freq': target_freq,
                    'dsp_freq_policy': 'manual', 'dsp_freq': 0.,
                    'args': 'mode_n=integer'}, chan)
                self.assertIsInstance(tune_result, dict)


    def test_tx_freq(self):
        """Test methods to get/set/enumerate TX frequency."""
        for chan in range(self.dut.get_tx_num_channels()):
            self.helper_range(lambda x: self.dut.set_tx_freq(x, chan),
                              lambda: self.dut.get_tx_freq(chan),
                              lambda: self.dut.get_tx_freq_range(chan),
                              steps=10)
            r = self.dut.get_tx_freq_range(chan)
            target_freq = round((r['start'] + r['stop']) / 2. / 1.e9) * 1.e9
            for policy in ['none', 'auto', 'manual']:
                tune_result = self.dut.set_tx_freq({'target_freq': target_freq,
                    'lo_off': 0., 'rf_freq_policy': 'manual', 'rf_freq': target_freq,
                    'dsp_freq_policy': 'manual', 'dsp_freq': 0.,
                    'args': 'mode_n=integer'}, chan)
                self.assertIsInstance(tune_result, dict)

    def test_rx_bandwidth(self):
        """Test methods to get/set/enumerate RX bandwidth."""
        for chan in range(self.dut.get_rx_num_channels()):
            self.helper_range(lambda x: self.dut.set_rx_bandwidth(x, chan),
                              lambda: self.dut.get_rx_bandwidth(chan),
                              lambda: self.dut.get_rx_bandwidth_range(chan),
                              steps=10)

    def test_tx_bandwidth(self):
        """Test methods to get/set/enumerate TX bandwidth."""
        for chan in range(self.dut.get_tx_num_channels()):
            self.helper_range(lambda x: self.dut.set_tx_bandwidth(x, chan),
                              lambda: self.dut.get_tx_bandwidth(chan),
                              lambda: self.dut.get_tx_bandwidth_range(chan),
                              steps=10)

    def test_rx_gain(self):
        """Test methods to get/set/enumerate RX gain."""
        for chan in range(self.dut.get_rx_num_channels()):
            self.helper_range(lambda x: self.dut.set_rx_gain(x, chan),
                              lambda: self.dut.get_rx_gain(chan),
                              lambda: self.dut.get_rx_gain_range(chan))

    def test_tx_gain(self):
        """Test methods to get/set/enumerate TX gain."""
        for chan in range(self.dut.get_tx_num_channels()):
            self.helper_range(lambda x: self.dut.set_tx_gain(x, chan),
                              lambda: self.dut.get_tx_gain(chan),
                              lambda: self.dut.get_tx_gain_range(chan))

    def test_rx_antenna(self):
        """Test methods to get/set/enumerate RX antenna mode."""
        for chan in range(self.dut.get_rx_num_channels()):
            self.helper_range(lambda x: self.dut.set_rx_antenna(x, chan),
                              lambda: self.dut.get_rx_antenna(chan),
                              lambda: self.dut.get_rx_antennas(chan))

    def test_tx_antenna(self):
        """Test methods to get/set/enumerate TX antenna mode."""
        for chan in range(self.dut.get_tx_num_channels()):
            self.helper_range(lambda x: self.dut.set_tx_antenna(x, chan),
                              lambda: self.dut.get_tx_antenna(chan),
                              lambda: self.dut.get_tx_antennas(chan))

    def test_rx_rate(self):
        """Test methods to get/set RX rate: test quotients of
        master-clock-rate and 2^N."""
        mcr = self.dut.get_master_clock_rate()
        for chan in range(self.dut.get_rx_num_channels()):
            rates = self.dut.get_rx_rates(chan)
            self.assertIsInstance(rates, dict)
            self.helper_range(lambda x: self.dut.set_rx_rate(x, chan),
                              lambda: self.dut.get_rx_rate(chan),
                              lambda: [mcr / 2.**n for n in range(5)])

    def test_tx_rate(self):
        """Test methods to get/set TX rate: test quotients of
        master-clock-rate and 2^N."""
        mcr = self.dut.get_master_clock_rate()
        for chan in range(self.dut.get_tx_num_channels()):
            rates = self.dut.get_rx_rates(chan)
            self.assertIsInstance(rates, dict)
            self.helper_range(lambda x: self.dut.set_tx_rate(x, chan),
                              lambda: self.dut.get_tx_rate(chan),
                              lambda: [mcr / 2.**n for n in range(5)])

    def test_tx_subdev_spec(self):
        """Test methods to set/get TX sub-device specification."""
        subdev = self.dut.get_tx_subdev_spec(0)
        self.assertTrue(isinstance(subdev, str) and len(subdev))
        self.dut.set_tx_subdev_spec(subdev, 0)

    def test_rx_subdev_spec(self):
        """Test methods to set/get RX sub-device specification."""
        subdev = self.dut.get_rx_subdev_spec(0)
        self.assertTrue(isinstance(subdev, str) and len(subdev))
        self.dut.set_rx_subdev_spec(subdev, 0)

    def test_tx_getters(self):
        """Test TX-specific getters."""
        num_tx = self.dut.get_tx_num_channels()
        self.assertTrue(isinstance(num_tx, int) and num_tx >= 0)
        for chan in range(num_tx):
            # get_fe_tx_freq_range
            fe_range = self.dut.get_fe_tx_freq_range(chan)
            self.assertTrue(is_range(fe_range))
            # get_tx_gain_names
            names = self.dut.get_tx_gain_names(chan)
            self.assertTrue(all(isinstance(n, str) and len(n) for n in names))
            # get_tx_sensor_names
            names = self.dut.get_tx_sensor_names(chan)
            self.assertTrue(all(isinstance(n, str) and len(n) for n in names))
            # get_tx_subdev_name
            subdev = self.dut.get_tx_subdev_name(chan)
            self.assertTrue(isinstance(subdev, str) and len(subdev))
            # get_usrp_tx_info
            info = self.dut.get_usrp_tx_info(chan)
            self.assertIsInstance(info, dict)

    def test_rx_getters(self):
        """Test RX-specific getters."""
        num_rx = self.dut.get_rx_num_channels()
        self.assertTrue(isinstance(num_rx, int) and num_rx >= 0)
        for chan in range(num_rx):
            # get_fe_rx_freq_range
            fe_range = self.dut.get_fe_rx_freq_range(chan)
            self.assertTrue(is_range(fe_range))
            # get_rx_gain_names
            names = self.dut.get_rx_gain_names(chan)
            self.assertTrue(all(isinstance(n, str) and len(n) for n in names))
            # get_rx_sensor_names
            names = self.dut.get_rx_sensor_names(chan)
            self.assertTrue(all(isinstance(n, str) and len(n) for n in names))
            # get_rx_subdev_name
            subdev = self.dut.get_rx_subdev_name(chan)
            self.assertTrue(isinstance(subdev, str) and len(subdev))
            # get_usrp_rx_info
            info = self.dut.get_usrp_rx_info(chan)
            self.assertIsInstance(info, dict)

    def test_mboard_getters(self):
        """ Test mainboard-specific getters."""
        # get_mboard_name
        mboard = self.dut.get_mboard_name(0)
        self.assertTrue(isinstance(mboard, str) and len(mboard))
        # get_mboard_sensor_names
        sensors = self.dut.get_mboard_sensor_names(0)
        self.assertTrue(all(isinstance(n, str) and len(n) for n in sensors))
        # get_num_mboards
        num_mboards = self.dut.get_num_mboards()
        self.assertTrue(isinstance(num_mboards, int) and num_mboards > 0)
        # get_pp_string
        pp_string = self.dut.get_pp_string()
        self.assertTrue(isinstance(pp_string, str) and len(pp_string))
        # get_time_synchronized
        sync = self.dut.get_time_synchronized()
        self.assertTrue(isinstance(sync, bool) and sync)
        # get_filter_names
        if hasattr(self.dut, 'get_filter_names'):
            filters = self.dut.get_filter_names()
            self.assertTrue(all(isinstance(n, str) and len(n) for n in filters))
        else:
            # UHD v4.x.x does not have get_filter_names
            self.assertTrue(int(uhd.UHD_VERSION_ABI.split('.')[0]) >= 4,
                            'Expected UHD major version >= 4.')

    def test_rx_setters(self):
        """Test RX-specific setters."""
        for chan in range(self.dut.get_rx_num_channels()):
            # set_rx_agc
            for value in [True, False]:
                self.dut.set_rx_agc(value, chan)
            # set_rx_dc_offset
            for value in [True, False]:
                self.dut.set_rx_dc_offset(value, chan)
            self.dut.set_rx_dc_offset(0. + 0j, chan)
            # set_rx_iq_balance
            for value in [True, False]:
                self.dut.set_rx_iq_balance(value, chan)
            self.dut.set_rx_iq_balance(0. + 0j, chan)

    def test_tx_setters(self):
        """Test TX-specific setters."""
        for chan in range(self.dut.get_tx_num_channels()):
            # set_tx_dc_offset
            self.dut.set_tx_dc_offset(0. + 0j, chan)
            # set_tx_iq_balance
            self.dut.set_tx_iq_balance(0. + 0j, chan)

    def test_mboard_setters(self):
        """Test mainboard-specific setters."""
        self.dut.clear_command_time(0)

    def test_gpio(self):
        """Test GPIO interface methods.
        Possible GPIO attribute names:
          - CTRL - 1 for ATR mode 0 for GPIO mode
          - DDR - 1 for output 0 for input
          - OUT - GPIO output level (not ATR mode)
          - ATR_0X - ATR idle state
          - ATR_RX - ATR receive only state
          - ATR_TX - ATR transmit only state
          - ATR_XX - ATR full duplex state
          - READBACK - readback input GPIOs"""
        attrs = ['CTRL', 'DDR', 'OUT', 'ATR_0X', 'ATR_RX',
                 'ATR_TX', 'ATR_XX', 'READBACK']
        banks = self.dut.get_gpio_banks(0)
        self.assertIsInstance(banks, list)
        self.assertTrue(all(isinstance(n, str) and len(n) for n in banks))
        # Test bank 'FP0' only, all devices should have this
        self.assertIn('FP0', banks)
        bank = 'FP0'
        for attr in attrs:
            value = self.dut.get_gpio_attr(bank, attr, 0)
            self.dut.set_gpio_attr(bank, attr, value, 0xFFFFFFFF, 0)

    def test_receive(self):
        """Test receive methods."""
        self.dut.set_master_clock_rate(16.e6)
        self.dut.set_rx_rate(1.e6)
        channels = list(range(self.dut.get_rx_num_channels()))
        channels_cases = [[x for x in c if x is not None] for c in combinations(
                          channels + [None] * (len(channels) - 1), len(channels))]
        channels_types = (list, tuple)
        kwargs_cases = (
            {},
            {'seconds_in_future': 1.0},
            {'timeout': 0.5},
            {'streaming': False},
            ({'streaming': True}, {}),
            ({'streaming': True, 'recycle': False}, {}),
            ({'streaming': True, 'recycle': True}, {}),
            ({'streaming': True, 'recycle': True}, {'fresh': False}),
            ({'streaming': True, 'recycle': True}, {'fresh': True}),
            {'otw_format': 'sc16'},
            {'otw_format': 'sc8'},
        )
        for channels_type, channels_case in product(channels_types, channels_cases):
            channels = channels_type(channels_case)
            for num_samps in (100, 2**12, 2**16, 2**20):
                for kwargs in kwargs_cases:
                    print('Testing RX (channels = {}, num_samps = {}, kwargs = {})'.format(
                          channels, num_samps, kwargs))
                    try:
                        if isinstance(kwargs, tuple):
                            self.dut.receive(num_samps, channels, **kwargs[0])
                            samps = self.dut.receive(**kwargs[1])
                            self.dut.stop_receive()
                        else:
                            samps = self.dut.receive(num_samps, channels, **kwargs)
                        self.assertEqual(len(samps), len(channels))
                        self.assertTrue(all(len(i) == num_samps for i in samps))
                    except uhd.UhdError as e:
                        self.fail('Failed to receive (channels = {}, num_samps = {}, kwargs'
                                  ' = {}): {}'.format(channels, num_samps, kwargs, str(e)))

    def test_transmit(self):
        """Test transmit methods."""
        self.dut.set_master_clock_rate(16.e6)
        self.dut.set_tx_rate(1.e6)
        channels = list(range(self.dut.get_tx_num_channels()))
        channels_cases = [[x for x in c if x is not None] for c in combinations(
                          channels + [None] * (len(channels) - 1), len(channels))]
        channels_types = (list, tuple)
        kwargs_cases = (
            {},
            {'seconds_in_future': 1.0},
            {'timeout': 0.5},
            {'continuous': False},
            {'continuous': True},
            {'otw_format': 'sc16'},
            {'otw_format': 'sc8'},
        )
        for channels_type, channels_case in product(channels_types, channels_cases):
            channels = channels_type(channels_case)
            for num_samps in (2**12, 2**16, 2**20):
                samps = [np.zeros((num_samps,), dtype=np.complex64) for _ in channels]
                for kwargs in kwargs_cases:
                    print('Testing TX (channels = {}, num_samps = {}, kwargs = {})'.format(
                          channels, num_samps, kwargs))
                    try:
                        self.dut.transmit(samps, channels, **kwargs)
                        if kwargs.get('continuous', False):
                            self.dut.stop_transmit()
                        sleep(100e-3)
                    except uhd.UhdError as e:
                        self.fail('Failed to transmit (channels = {}, num_samps = {}, kwargs'
                                  ' = {}): {}'.format(channels, num_samps, kwargs, str(e)))


if __name__ == '__main__':
    unittest.main()
