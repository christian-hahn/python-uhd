#!/usr/bin/env python3

"""Implements a spectrum analyzer.
Tested with USRP B210 & B200mini."""

from pyuhd import Usrp
import signal
import sys
import numpy as np
import argparse

from spec_an import SpecAn


def main():
    """Entry point. Contains main loop."""

    parser = argparse.ArgumentParser()
    parser.add_argument("center_freq", help="Center frequency in Hz.", type=float)
    parser.add_argument("span", help="Span in Hz.", type=float)
    parser.add_argument("--gain", help="Gain in dB.", type=float, default=40.0)
    parser.add_argument(
        "--channels", help="Channels.", type=int, nargs="+", default=[0]
    )
    parser.add_argument(
        "--num-samps", help="Number of samples per capture.", type=int, default=2**18
    )
    parser.add_argument(
        "--num-points", help="Number of samples in PSD.", type=int, default=2**12
    )
    args = parser.parse_args()

    # Create USRP object
    u = Usrp()

    # Register an interrupt handler to catch Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Set the master clock rate, then set the sample rate derived from an
    # even integer divisor of the master clock rate
    master_clock_rate = 30.72e6 if len(args.channels) > 1 else 61.44e6
    master_clock_rate = float(
        args.span * np.floor(master_clock_rate / args.span / 2.0) * 2.0
    )
    u.set_master_clock_rate(master_clock_rate)
    master_clock_rate = u.get_master_clock_rate()
    divisor = max(float(np.floor(master_clock_rate / args.span / 2.0) * 2.0), 1)
    u.set_rx_rate(u.get_master_clock_rate() / divisor)

    # u.set_rx_rate(args.span)
    samp_rate = u.get_rx_rate()

    # Setup each channel: set gain, LO frequency, etc.
    for chan in args.channels:
        print("[RX channel {}]:".format(chan))
        u.set_rx_bandwidth(args.span, chan)
        tune_result = u.set_rx_freq(args.center_freq, chan)
        print("  tune-result =")
        for k, v in tune_result.items():
            print("    {} = {}".format(k, v))
        u.set_rx_antenna("RX2", chan)
        u.set_rx_gain(args.gain, chan)
        print("  bw = {} Hz".format(u.get_rx_bandwidth(chan)))
        print("  freq = {} Hz".format(u.get_rx_freq(chan)))
        print("  antenna = {}".format(u.get_rx_antenna(chan)))
        print("  gain = {:.3f} dB".format(u.get_rx_gain(chan)))
    print("master_clock_rate = {} Hz".format(u.get_master_clock_rate()))
    print("rx_rate = {} Hz".format(u.get_rx_rate()))

    spec = SpecAn(
        samp_rate,
        args.num_points,
        num_chans=len(args.channels),
        limits=[-125.0, 0.0],
        freq_offset=args.center_freq,
    )

    # Start streaming & update graphic continuously
    try:
        u.receive(
            args.num_samps,
            args.channels,
            streaming=True,
            recycle=True,
        )
        while True:
            # Receive a block of samples
            samps, _ = u.receive()
            # Compute average and peak power
            samps_sqrd = np.real(samps[0] * np.conj(samps[0]))
            avg_pwr = 10.0 * np.log10(np.mean(samps_sqrd))
            peak_pwr = 10.0 * np.log10(np.max(samps_sqrd))
            print(
                "avg-pwr = {:.3f} dBfs, peak-pwr = {:.3f} dBfs".format(
                    avg_pwr, peak_pwr
                )
            )
            spec.update(samps)
    finally:
        u.stop_receive()


def signal_handler(signal, frame):
    """Interrupt handler to catch Ctrl+C and exit."""
    sys.exit(0)


if __name__ == "__main__":
    main()
