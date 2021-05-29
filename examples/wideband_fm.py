#!/usr/bin/env python3

''' Implements a wideband FM receiver, outputs audio using alsaaudio.
    Tested with USRP B210 & B200mini.

    Requires alsaaudio. To install under Ubuntu, do:

    sudo apt-get install libasound2-dev
    sudo pip3 install pyalsaaudio
'''

from pyuhd import Usrp
import numpy as np
from scipy.signal import lfilter, cheby1
import alsaaudio
import argparse


def main():
    """ Entry point. Contains main loop. """

    parser = argparse.ArgumentParser()
    parser.add_argument('signal_freq', help='Signal frequency in Hz.',
                        type=float)
    parser.add_argument('--gain', help='RX gain in dB.', type=float, default=40.)
    parser.add_argument('--max-deviation', help='Maximum deviation in Hz.',
                        type=float, default=75.e3)
    args = parser.parse_args()

    # Constants
    channel = 0  # This is the receiver channel that will be used
    signal_bw = 200e3  # FM broadcast has a bandwidth of ~200 kHz
    audio_bw = 15.e3  # Audio bandwidth
    audio_samp_rate = 48e3  # Output audio sample-rate of 48 kSps

    # Create USRP object
    u = Usrp()

    # Select optimal LO frequency: signal_freq - bandwidth rounded
    # to nearest 1.25 MHz
    lo_freq = float(np.floor((args.signal_freq - signal_bw) / 1.25e6) * 1.25e6)

    # Set the LO frequency: round down to nearest achievable
    tune_result = u.set_rx_freq(lo_freq, channel)
    lo_freq = tune_result['actual_rf_freq']
    u.set_rx_freq(lo_freq, channel)
    lo_freq = u.get_rx_freq(channel)

    # Compute ideal sample-rates & bandwidths
    min_samp_rate = float((abs(args.signal_freq - lo_freq) + signal_bw) * 2.)
    if_samp_rate = float(np.ceil(signal_bw / audio_samp_rate) * audio_samp_rate)
    samp_rate = float(np.ceil(min_samp_rate / if_samp_rate) * if_samp_rate)
    master_clock_rate = float(samp_rate * np.floor(61.44e6 / samp_rate / 4.) * 4.)

    # Set the master clock rate
    u.set_master_clock_rate(master_clock_rate)
    master_clock_rate = u.get_master_clock_rate()

    # Set the sample rate
    u.set_rx_rate(samp_rate)
    samp_rate = u.get_rx_rate()

    # Compute the decimation factor and actual audio sample rate
    if_decim_factor = round(samp_rate / if_samp_rate)
    if_samp_rate = samp_rate / if_decim_factor
    audio_decim_factor = round(if_samp_rate / audio_samp_rate)
    audio_samp_rate = if_samp_rate / audio_decim_factor

    # Setup channel: set analog-bandwidth, antenna, gain
    u.set_rx_bandwidth(min_samp_rate, channel)
    u.set_rx_antenna('RX2', channel)
    u.set_rx_gain(args.gain, channel)

    # Compute the number of samples
    audio_num_samps = int(round(audio_samp_rate * 5.))  # 5 second blocks
    num_samps = ((audio_num_samps * audio_decim_factor) + 1) * if_decim_factor

    # Open sound device in playback mode
    out = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NONBLOCK)
    out.setchannels(1)
    out.setrate(int(round(audio_samp_rate)))
    out.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    out.setperiodsize(audio_num_samps)

    print('bandwidth = {} Hz'.format(u.get_rx_bandwidth(channel)))
    print('freq = {} Hz'.format(u.get_rx_freq(channel)))
    print('antenna = {}'.format(u.get_rx_antenna(channel)))
    print('gain = {:.3f} dB'.format(u.get_rx_gain(channel)))
    print('master_clock_rate = {} Hz'.format(u.get_master_clock_rate()))
    print('rx_rate = {} Hz'.format(u.get_rx_rate()))
    print('if_decim_factor = {}'.format(if_decim_factor))
    print('if_samp_rate = {} Hz'.format(if_samp_rate))
    print('audio_decim_factor = {}'.format(audio_decim_factor))
    print('audio_samp_rate = {} Hz'.format(audio_samp_rate))

    # Downconversion
    dnconv = np.exp(-1j * 2. * np.pi * ((args.signal_freq - lo_freq) / samp_rate)
             * np.arange(num_samps))

    # IF and audio low-pass filters
    if_filter = cheby1(N=8, rp=3., Wn=signal_bw / samp_rate, btype='low')
    audio_filter = cheby1(N=8, rp=3., Wn=15.e3 / if_samp_rate, btype='low')

    # De-emphasis filter
    decay = np.exp(-1. / (if_samp_rate * 75e-6))
    deemphasis_filter = [1 - decay], [1, -decay]

    try:
        # Start receive
        u.receive(num_samps, [channel], True)

        while True:

            # Get received samples
            samps = u.receive()[0]

            # Compute average power: this is for display purposes only
            samps_sqrd = np.real(np.conj(samps) * samps)
            avg_pwr = 10. * np.log10(np.mean(samps_sqrd))
            peak_pwr = 10. * np.log10(np.max(samps_sqrd))
            print('avg_pwr = {:.3f} dBfs, peak_pwr = {:.3f} dBfs'.format(avg_pwr, peak_pwr))

            # Downconvert to baseband
            samps = samps * dnconv

            # Low-pass filter + decimate
            samps = lfilter(*if_filter, samps)
            samps = samps[::if_decim_factor]

            # Phase-discriminator
            samps = np.angle(samps[1:] * np.conj(samps[:-1]))

            # De-emphasis filter, low-pass filter
            samps = lfilter(*audio_filter, samps)
            samps = lfilter(*deemphasis_filter, samps)

            # Decimate to audio-sample rate and scale samples
            # based on max-deviation
            samps = samps[::audio_decim_factor]
            samps = samps * (2.**14 / (args.max_deviation / if_samp_rate
                    * (2. * np.pi)))

            out.write(samps.astype('int16'))
    finally:
        u.stop_receive()


if __name__ == '__main__':
    main()
