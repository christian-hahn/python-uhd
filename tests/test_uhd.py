#!/usr/bin/env python3

import uhd
import unittest
import itertools as it

class UhdTestCase(unittest.TestCase):

    def setUp(self):
        self.dut = uhd.Uhd()

        center_freq = 100.e6
        self.channels = range(self.dut.get_rx_num_channels())

        samp_rate = self.dut.get_master_clock_rate()/16.
        self.dut.set_rx_rate(samp_rate)
        print('master_clock_rate = {} Hz'.format(self.dut.get_master_clock_rate()))
        print('rx_rate = {} Hz'.format(self.dut.get_rx_rate()))

        samp_rate = self.dut.get_rx_rate()
        for chan in self.channels:
            self.dut.set_rx_bandwidth(samp_rate, chan)
            self.dut.set_rx_freq(center_freq, chan)
            self.dut.set_rx_antenna('RX2', chan)
            self.dut.set_rx_gain(40., chan)
            print('{}:'.format(chan))
            print('  bw = {} Hz'.format(self.dut.get_rx_bandwidth(chan)))
            print('  freq = {} Hz'.format(self.dut.get_rx_freq(chan)))
            print('  antenna = {}'.format(self.dut.get_rx_antenna(chan)))
            print('  gain = {} dB'.format(self.dut.get_rx_gain(chan)))


    def tearDown(self):
        del self.dut

    def test_receive(self):
        for channels in [list(set(c)) for c in it.combinations_with_replacement(self.channels,2)]:
            for num_samps in [100, 2**12, 2**16, 2**20]:
                for streaming in [True, False]:
                    try:
                        print('Testing RX (channels = {}, num_samps = {}, streaming = {})'.format(
                              str(channels), num_samps, str(streaming)))
                        if streaming:
                            self.dut.receive(num_samps, channels, True)
                            samps = self.dut.receive()
                            self.dut.stop_receive()
                        else:
                            samps = self.dut.receive(num_samps, list(channels))
                            samps = self.dut.receive(num_samps, channels)
                        assert len(samps) == len(channels)
                        assert all(len(i) == num_samps for i in samps)
                    except uhd.UhdError as e:
                        self.fail('Failed to receive (channels = %s, num_samps = %d): %s' %
                                  (str(channels), num_samps, e.message))


if __name__ == '__main__':
    unittest.main()
