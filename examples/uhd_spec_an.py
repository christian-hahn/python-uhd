#!/usr/bin/env python3

''' Implements a spectrum analyzer.
    Tested with USRP B210 & B200mini. '''

import signal
import sys
import numpy as np
import argparse

from spec_an import SpecAn
import uhd


def signal_handler(signal, frame):
    """ Interrupt handler to catch Ctrl+C and shutdown receiver. """
    try:
        u.stop_receive()
    finally:
        sys.exit(0)


def main():
    """ Entry point. Contains main loop. """

    parser = argparse.ArgumentParser()
    parser.add_argument('center_freq', help='Center frequency in Hz.',
                        type=float)
    parser.add_argument('span', help='Span in Hz.', type=float)
    parser.add_argument('--gain', help='Gain in dB.', type=float, default=40.)
    parser.add_argument('--channels', help='Channels.', type=list, default=[0])
    parser.add_argument('--num-samps', help='Number of samples per capture.',
                        type=int, default=2**18)
    parser.add_argument('--num-points', help='Number of samples in PSD.',
                        type=int, default=2**14)
    args = parser.parse_args()

    # Create UHD object
    u = uhd.Uhd()

    # Register an interrupt handler to catch Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Set the master clock rate, then set the sample rate derived from an
    # even integer divisor of the master clock rate
    master_clock_rate = float(args.span * np.floor(61.44e6 / args.span / 4.) * 4.)
    u.set_master_clock_rate(master_clock_rate)
    master_clock_rate = u.get_master_clock_rate()
    divisor = max(float(np.floor(master_clock_rate / args.span / 2.) * 2.), 1)
    u.set_rx_rate(u.get_master_clock_rate() / divisor)
    samp_rate = u.get_rx_rate()

    # Setup each channel: set gain, LO frequency, etc.
    for chan in args.channels:
        print('[RX channel {}]:'.format(chan))
        u.set_rx_bandwidth(args.span, chan)
        tune_result = u.set_rx_freq(args.center_freq, chan)
        print('  tune-result =')
        for k,v in tune_result.items():
            print('    {} = {}'.format(k,v))
        u.set_rx_antenna('RX2', chan)
        u.set_rx_gain(args.gain, chan)
        print('  bw = {} Hz'.format(u.get_rx_bandwidth(chan)))
        print('  freq = {} Hz'.format(u.get_rx_freq(chan)))
        print('  antenna = {}'.format(u.get_rx_antenna(chan)))
        print('  gain = {:.3f} dB'.format(u.get_rx_gain(chan)))
    print('master_clock_rate = {} Hz'.format(u.get_master_clock_rate()))
    print('rx_rate = {} Hz'.format(u.get_rx_rate()))

    spec = SpecAn(samp_rate, args.num_points, num_chans=len(args.channels),
                  limits=[-125.,0.], freq_offset=args.center_freq)

    # Start streaming & update graphic continuously
    u.receive(args.num_samps, args.channels, True)
    while True:
        samps = u.receive()
        samps_sqrd = np.real(samps[0] * np.conj(samps[0]))
        avg_pwr = 10.*np.log10(np.mean(samps_sqrd))
        peak_pwr = 10.*np.log10(np.max(samps_sqrd))
        print('avg-pwr = {:.3f} dBfs, peak-pwr = {:.3f} dBfs'.format(avg_pwr, peak_pwr))
        spec.update(samps)


if __name__ == '__main__':
    main()
