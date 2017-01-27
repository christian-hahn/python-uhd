#!/usr/bin/env python2

import signal
import sys
import numpy as np

import spec_an
import uhd

u = uhd.Uhd()

# Parameters
center_freq = 154.0e6
span = 9.e6

u.set_master_clock_rate(32.e6)
divisor = float(np.floor((u.get_master_clock_rate()/span)/4.)*4.)
u.set_rx_rate(u.get_master_clock_rate()/divisor)
samp_rate = u.get_rx_rate()

channels = [0]
num_samps = 2**17
num_points = 2**14

u.set_rx_rate(samp_rate)
samp_rate = u.get_rx_rate()
for chan in channels:
    print '[RX channel %d]:' % (chan)
    u.set_rx_bandwidth(samp_rate, chan)
    u.set_rx_freq(center_freq, chan)
    u.set_rx_antenna('RX2', chan)
    u.set_rx_gain(80., chan)
    print '  bw = %f Hz' % (u.get_rx_bandwidth(chan))
    print '  freq = %f Hz' % (u.get_rx_freq(chan))
    print '  antenna = %s' % (u.get_rx_antenna(chan))
    print '  gain = %.3f dB' % (u.get_rx_gain(chan))
print 'master_clock_rate = %f Hz' % (u.get_master_clock_rate())
print 'rx_rate = %f Hz' % (u.get_rx_rate())

spec = spec_an.SpecAn(samp_rate, num_points, num_chans=len(channels), limits=[-100.,0.],
                      freq_offset=center_freq)

# Register an interrupt handler to catch Ctrl+C
def signal_handler(signal, frame):
    u.stop_receive()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# Start streaming & update graphic continuously
u.receive(num_samps, channels, True)
while True:
    samps = u.receive()
    spec.update(samps)
