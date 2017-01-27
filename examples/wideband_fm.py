import uhd
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from scipy import signal as scipy
import alsaaudio
import signal
import sys

u = uhd.Uhd()

def main():

    fS = 61440000.047871/64.
    signal_freq = 460.150000e6
    center_freq = signal_freq - .275e6
    channels = [0]

    u.set_rx_rate(fS)
    fS = u.get_rx_rate()
    for chan in channels:
        u.set_rx_bandwidth(fS, chan)
        u.set_rx_freq(center_freq, chan)
        u.set_rx_antenna('RX2', chan)
        u.set_rx_gain(80., chan)
        print '%d:' % (chan)
        print ' bw = %f Hz' % (u.get_rx_bandwidth(chan))
        print ' freq = %f Hz' % (u.get_rx_freq(chan))
        print ' antenna = %s' % (u.get_rx_antenna(chan))
        print ' gain = %.3f dB' % (u.get_rx_gain(chan))

    print 'master_clock_rate = %f Hz' % (u.get_master_clock_rate())
    print 'rx_rate = %f Hz' % (u.get_rx_rate())

    num_samps = int(fS*1)

    actual_center_freq = u.get_rx_freq(0)
    fS = u.get_rx_rate()
    f_tone = signal_freq - actual_center_freq

    ref_wfm = np.exp(-1j*2*np.pi*(f_tone/fS)*np.arange(num_samps))

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
