#!/usr/bin/env python3

''' Implements a wideband FM receiver, outputs audio using alsaaudio.
    Tested with USRP B210 & B200mini.

    Requires alsaaudio. To install under Ubuntu, do:

    sudo apt-get install libasound2-dev
    sudo pip3 install pyalsaaudio
'''

import uhd
import numpy as np
from scipy import signal as scipy
import alsaaudio
import signal
import sys

def main():

    # Parameters
    signal_freq = 99.7e6  # This is the carrier frequency
    chan = 0  # This is the receiver channel that will be used
    gain = 40.  # Receiver gain
    signal_bw = 200000.  # FM broadcast has a bandwidth of ~200 kHz
    master_clock_rate = 32.e6

    # Create UHD object
    u = uhd.Uhd()

    # Register an interrupt handler to catch Ctrl+C
    def signal_handler(signal, frame):
        try:
            u.stop_receive()
        finally:
            sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # Set the master clock rate
    u.set_master_clock_rate(master_clock_rate)
    master_clock_rate = u.get_master_clock_rate()

    # Set the LO frequency: round down to nearest achievable
    center_freq = signal_freq - signal_bw  # This is the receiver LO frequency
    tune_result = u.set_rx_freq(center_freq, chan)
    print('tune-result =')
    for k,v in tune_result.items():
        print('  {} = {}'.format(k,v))
    center_freq = tune_result['actual_rf_freq']
    u.set_rx_freq(center_freq, chan)
    center_freq = u.get_rx_freq(chan)

    # Set the sample rate derived from an even integer divisor
    # of the master clock rate
    samp_rate = signal_freq - center_freq + signal_bw*2.
    divisor = max(float(np.floor(master_clock_rate/samp_rate/2.)*2.), 2)
    u.set_rx_rate(u.get_master_clock_rate()/divisor)
    samp_rate = u.get_rx_rate()

    # Setup channel: set analog-bandwidth, antenna, gain
    u.set_rx_bandwidth(samp_rate, chan)
    u.set_rx_antenna('RX2', chan)
    u.set_rx_gain(gain, chan)

    print('bandwidth = {} Hz'.format(u.get_rx_bandwidth(chan)))
    print('freq = {} Hz'.format(u.get_rx_freq(chan)))
    print('antenna = {}'.format(u.get_rx_antenna(chan)))
    print('gain = {:.3f} dB'.format(u.get_rx_gain(chan)))
    print('master_clock_rate = {} Hz'.format(u.get_master_clock_rate()))
    print('rx_rate = {} Hz'.format(u.get_rx_rate()))

    # Number of samples: round down to nearest 2^N
    num_samps = int(2**np.floor(np.log2(samp_rate*1.)))

    f_tone = signal_freq - center_freq
    ref_wfm = np.exp(-1j*2*np.pi*(f_tone/samp_rate)*np.arange(num_samps))

    # Start receive
    u.receive(num_samps, [chan], True)

    first = True
    max_sig_audio = 0.
    while True:

        # Get received samples
        samps = u.receive()

        # Compute average power: this is for display purposes only
        samps_sqrd = np.real(np.conj(samps[0])*samps[0])
        avg_pwr = 10.*np.log10(np.mean(samps_sqrd))
        peak_pwr = 10.*np.log10(np.max(samps_sqrd))
        print('avg_pwr = {:.3f} dBfs, peak_pwr = {:.3f} dBfs'.format(avg_pwr, peak_pwr))

        # Down-convert to baseband
        samps_dwn = samps[0]*ref_wfm

        # Low-pass filter & decimate
        dec_rate = int(samp_rate / signal_bw)
        samps_dec = scipy.decimate(samps_dwn, dec_rate, zero_phase=False)
        samp_rate_dec = samp_rate/dec_rate

        # Polar discriminator
        polar_desc = np.angle(samps_dec[1:] * np.conj(samps_dec[:-1]))

        # Apply de-emphasis filter
        x = np.exp(-1/(samp_rate_dec * 75e-6))   # Calculate the decay between each sample
        sig_deemp = scipy.lfilter([1-x],[1,-x],polar_desc)

        # Decimate signal to audio sampling rate: ~44-48 kHz
        samp_rate_audio = 44100.0
        dec_rate_audio = int(samp_rate_dec/samp_rate_audio)
        samp_rate_audio = samp_rate_dec / dec_rate_audio
        sig_audio = scipy.decimate(sig_deemp, dec_rate_audio, zero_phase=False)

        # Scale audio signal: scale audio signal accordingn to max seen
        max_sig_audio = max(float(np.max(np.abs(sig_audio))), max_sig_audio)
        if max_sig_audio > 0.:
            sig_audio = sig_audio * (2.**14 / max_sig_audio)

        if first:
            print('samp_rate_audio = {} Hz'.format(samp_rate_audio))

            # Open the device in playback mode.
            out = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NONBLOCK)
            out.setchannels(1)
            out.setrate(int(round(samp_rate_audio)))
            out.setformat(alsaaudio.PCM_FORMAT_S16_LE)

            # The period size controls the internal number of frames per period.
            # The significance of this parameter is documented in the ALSA api.
            out.setperiodsize(len(sig_audio))

        out.write(sig_audio.astype('int16'))
        first = False

    u.stop_receive()


if __name__ == '__main__':
    main()
