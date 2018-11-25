#!/usr/bin/env python3

import uhd

u = uhd.Uhd()

# Parameters
center_freq = 140.625e6
samp_rate = 1.0e6
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

"""(1) Capture samples: not streaming"""
samps = u.receive(
    num_samps,
    channels,
    streaming=False,
    seconds_in_future=1.0,
    timeout=0.5,
)

"""(2) Capture samples: streaming without recycle"""
u.receive(
    num_samps,
    channels,
    streaming=True,
    recycle=False,
    seconds_in_future=1.0,
    timeout=0.5,
)
samps = u.receive()
u.stop_receive()

"""(3) Capture samples: streaming with recycle
       When recycle = True, 1 block of samples is always ready to be receive()'d.
       Old blocks of samples are discarding, and the underlying buffers are
       recycled to be used for the next, newer block of samples."""
u.receive(
    num_samps,
    channels,
    streaming=True,
    recycle=True,
    seconds_in_future=1.0,
    timeout=0.5,
)
"""Fresh = True guarantees that the time of the first sample is after the call
   to receive().  That the samples are indeed fresh, not stale. """
samps = u.receive(fresh=True)
u.stop_receive()
