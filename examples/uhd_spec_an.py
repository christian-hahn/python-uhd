#!/usr/bin/env python3

''' Implements a spectrum analyzer.
    Tested with USRP B210 & B200mini. '''

import signal
import sys
import numpy as np

from spec_an import SpecAn
import uhd

# Create UHD object
u = uhd.Uhd()

# Parameters
center_freq = 99.0e6
span = 9.e6
gain = 30.
channels = [0]
num_samps = 2**17
num_points = 2**14
master_clock_rate = 32.e6

# Register an interrupt handler to catch Ctrl+C
def signal_handler(signal, frame):
    try:
        u.stop_receive()
    finally:
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# Set the master clock rate, then set the sample rate derived from an
# even integer divisor of the master clock rate
u.set_master_clock_rate(master_clock_rate)
master_clock_rate = u.get_master_clock_rate()
divisor = max(float(np.floor(master_clock_rate/span/2.)*2.), 2)
u.set_rx_rate(u.get_master_clock_rate()/divisor)
samp_rate = u.get_rx_rate()

# Setup each channel: set gain, LO frequency, etc.
for chan in channels:
    print('[RX channel {}]:'.format(chan))
    u.set_rx_bandwidth(samp_rate, chan)
    tune_result = u.set_rx_freq(center_freq, chan)
    print('  tune-result =')
    for k,v in tune_result.items():
        print('    {} = {}'.format(k,v))
    u.set_rx_antenna('RX2', chan)
    u.set_rx_gain(gain, chan)
    print('  bw = {} Hz'.format(u.get_rx_bandwidth(chan)))
    print('  freq = {} Hz'.format(u.get_rx_freq(chan)))
    print('  antenna = {}'.format(u.get_rx_antenna(chan)))
    print('  gain = {:.3f} dB'.format(u.get_rx_gain(chan)))
print('master_clock_rate = {} Hz'.format(u.get_master_clock_rate()))
print('rx_rate = {} Hz'.format(u.get_rx_rate()))

spec = SpecAn(samp_rate, num_points, num_chans=len(channels), limits=[-125.,0.],
              freq_offset=center_freq)

# Start streaming & update graphic continuously
u.receive(num_samps, channels, True)
while True:
    samps = u.receive()
    samps_sqrd = np.real(samps[0]*np.conj(samps[0]))
    avg_pwr = 10.*np.log10(np.mean(samps_sqrd))
    peak_pwr = 10.*np.log10(np.max(samps_sqrd))
    print('avg_pwr = {:.3f} dBfs, peak_pwr = {:.3f} dBfs'.format(avg_pwr, peak_pwr))
    spec.update(samps)
