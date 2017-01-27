import uhd
import time
import datetime
import numpy as np
import cPickle as pickle
from scipy import signal as scipy
import alsaaudio
import signal
import sys

u = uhd.Uhd()

signal_freqs = [460.150e6, 460.275e6, 460.200e6, 460.425e6, 460.400e6, 460.325e6,
                460.475e6, 460.525e6, 460.100e6, 460.050e6, 460.175e6, 460.300e6]

signal_freqs = [460.150e6, 460.425e6]


# Register an interrupt handler to catch Ctrl+C
def signal_handler(signal, frame):
    u.stop_receive()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def main():

    center_freq = float(np.min(np.floor((np.array(signal_freqs)-100e3)/1e6))*1e6)
    min_samp_rate = float(np.max(np.array(signal_freqs)-center_freq)*2)
    divisor = float(2.**np.floor(np.log2(u.get_master_clock_rate()/(min_samp_rate*1.1))))
    u.set_rx_rate(u.get_master_clock_rate()/divisor)
    samp_rate = u.get_rx_rate()

    print 'center_freq: %f' % (center_freq)
    print 'min samp_rate: %f' % (min_samp_rate)
    print 'divisor: %f' % (divisor)
    print 'samp_rate: %f Hz' % (samp_rate)

    channels = [0]

    for chan in channels:
        u.set_rx_bandwidth(samp_rate, chan)
        u.set_rx_freq(center_freq, chan)
        u.set_rx_antenna('RX2', chan)
        u.set_rx_gain(60., chan)
        print '%d:' % (chan)
        print ' bw = %f Hz' % (u.get_rx_bandwidth(chan))
        print ' freq = %f Hz' % (u.get_rx_freq(chan))
        print ' antenna = %s' % (u.get_rx_antenna(chan))
        print ' gain = %.3f dB' % (u.get_rx_gain(chan))

    filt_num_taps = 100
    signal_bw = 7.e3
    filt_coeffs = scipy.firwin(filt_num_taps + 1, 1. / (samp_rate/signal_bw), window='hamming')

    decim_factor = int(np.round(samp_rate/44.e3))
    samp_rate_low = samp_rate/decim_factor

    num_samps = int(2.**np.ceil(np.log2(samp_rate*1.)))
    num_samps_low = np.floor(num_samps / decim_factor)

    dnconv = {}
    for signal_freq in signal_freqs:
        dnconv[signal_freq] = np.exp(-1j*2*np.pi*((signal_freq-center_freq)/samp_rate)*np.arange(num_samps))

    # Open the device in playback mode.
    out = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NONBLOCK)
    out.setchannels(1)
    out.setrate(int(round(samp_rate_low)))
    out.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    # The period size controls the internal number of frames per period.
    # The significance of this parameter is documented in the ALSA api.
    out.setperiodsize(int(num_samps_low))

    max_samps_desc = None

    u.receive(num_samps, channels, True)
    while True:
        samps = u.receive()[0]
        avg_pwr = 10.*np.log10(np.mean(np.conj(samps)*samps))
        peak_pwr = 10.*np.log10(np.max(np.abs((np.conj(samps)*samps))))
        print avg_pwr
        print peak_pwr

        samps_audio = []
        for signal_freq in signal_freqs:

            # Downconvert
            samps_dwn = samps*dnconv[signal_freq]

            # Filter
            samps_filt = scipy.lfilter(filt_coeffs, 1., samps_dwn, axis=0)

            # Filter & decimate
            samps_decim = samps_filt[::decim_factor]

            # polar discriminator
            samps_desc = np.angle(samps_decim[1:] * np.conj(samps_decim[:-1]))

            avg_pwr = np.convolve(np.conj(samps_decim[1:])*samps_decim[1:],np.ones(1000)/1000.,'same')

            print '(%.3f, %.3f)' % (10.*np.log10(np.min(avg_pwr)),10.*np.log10(np.max(avg_pwr)))

            samps_desc[avg_pwr < 10.**(-60./10.)] = 0

            # The de-emphasis filter
            x = np.exp(-1/(samp_rate_low * 75e-6))   # Calculate the decay between each sample
            samps_desc = scipy.lfilter([1-x],[1,-x],samps_desc)

            max_samps_desc = max(np.max(np.abs(samps_desc)), max_samps_desc)
            if np.max(np.abs(samps_desc)) > 0.:
                samps_desc = samps_desc * (10000. / max_samps_desc)

            if len(samps_audio):
                samps_audio = samps_audio + samps_desc
            else:
                samps_audio = samps_desc

        out.write(samps_audio.astype('int16'))


    exit()

    fS = 61440000.047871/64.
    signal_freq = 460.150000e6
    center_freq = signal_freq - .275e6






    actual_center_freq = u.get_rx_freq(0)
    fS = u.get_rx_rate()
    f_tone = signal_freq - actual_center_freq



    # Register an interrupt handler to catch Ctrl+C
    def signal_handler(signal, frame):
        u.stop_receive()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    u.receive(num_samps, channels, True)

    first = True
    while True:

        samps = u.receive()

        samps_dwn = samps[0]*ref_wfm

        # An FM broadcast signal has a bandwidth of 200 kHz
        f_bw = 20000.
        dec_rate = int(fS / f_bw)
        samps_dec = scipy.decimate(samps_dwn, dec_rate)
        # Calculate the new sampling rate
        fS_dec = fS/dec_rate

        # polar discriminator
        polar_desc = np.angle(samps_dec[1:] * np.conj(samps_dec[:-1]))

        print 10.*np.log10(np.mean( np.conj(samps_dec[1:])*samps_dec[1:] ))

        polar_desc[(np.conj(samps_dec[1:])*samps_dec[1:]) < 10.**(-45./10.)] = 0

        # The de-emphasis filter
        x = np.exp(-1/(fS_dec * 75e-6))   # Calculate the decay between each sample
        sig_deemp = scipy.lfilter([1-x],[1,-x],polar_desc)

        # Find a decimation rate to achieve audio sampling rate between 44-48 kHz
        audio_freq = f_bw#44100.0
        dec_audio = int(fS_dec/audio_freq)
        Fs_audio = fS_dec / dec_audio

        sig_audio = sig_deemp# scipy.decimate(sig_deemp, dec_audio)

        max_sig_audio = np.max(np.abs(sig_audio))
        if np.max(np.abs(sig_audio)) > 0.:
            sig_audio = sig_audio * (10000. / np.max(np.abs(sig_audio)))

        if first:
            # Open the device in playback mode.
            out = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NONBLOCK)
            out.setchannels(1)
            out.setrate(int(round(Fs_audio)))
            out.setformat(alsaaudio.PCM_FORMAT_S16_LE)

            # The period size controls the internal number of frames per period.
            # The significance of this parameter is documented in the ALSA api.
            out.setperiodsize(len(sig_audio))

        out.write(sig_audio.astype('int16'))

        first = False

    u.stop_receive()


if __name__ == '__main__':
    main()
