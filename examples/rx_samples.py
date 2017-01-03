#!/usr/bin/env python3

import uhd

u = uhd.Uhd()

# Parameters
center_freq = 140.625e6
samp_rate = 10.0e6
num_samps = 2**16
channels = range(u.get_rx_num_channels())

# Set sample rate
u.set_rx_rate(samp_rate)

# For each channel
for chan in channels:
    u.set_rx_bandwidth(samp_rate, chan)
    u.set_rx_freq(center_freq, chan)
    u.set_rx_antenna('RX2', chan)
    u.set_rx_gain(40., chan)

# Capture samples
samps = u.receive(num_samps, channels, False)
