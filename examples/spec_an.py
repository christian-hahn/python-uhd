"""Spectrum analyzer visualization

This module provides a PSD visualization. Uses matplotlib and Qt5Agg backend.

To install matplotlib using pip: pip3 install matplotlib
To install Qt5Agg using pip: pip3 install pyqt5

"""

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")


class SpecAn(object):

    def __init__(
        self,
        samp_rate,
        num_points,
        num_chans=1,
        title=None,
        limits=None,
        freq_offset=0.0,
    ):

        self.samp_rate = samp_rate
        self.num_points = num_points
        self.num_chans = num_chans
        self.limits = limits
        self.freq_offset = freq_offset

        plt.ion()

        self.fig, self.axes = plt.subplots()
        self.traces = []
        freqs = (
            np.fft.fftshift(
                signal.welch(
                    np.zeros(num_points),
                    samp_rate,
                    nperseg=num_points,
                    scaling="spectrum",
                    return_onesided=False,
                )[0]
            )
            + freq_offset
        )
        for chan in range(num_chans):
            self.traces.append(self.axes.plot(freqs, np.ma.masked_all_like(freqs))[0])
        self.axes.grid()

        if title:
            self.axes.set_title(title)
        self.axes.set_xlabel("Frequency, MHz")
        self.axes.set_ylabel("Magnitude, dBm")
        self.axes.set_xlim([min(freqs), max(freqs)])
        if limits:
            self.axes.set_ylim(limits)
        self.fig.show()
        self.fig.canvas.draw()

    def update(self, samps):
        ymin = None
        ymax = None
        for samp, trace in zip(samps, self.traces):
            psd = 10.0 * np.log10(
                np.fft.fftshift(
                    signal.welch(
                        samp,
                        self.samp_rate,
                        nperseg=self.num_points,
                        scaling="spectrum",
                        return_onesided=False,
                    )[1]
                )
            )
            trace.set_ydata(psd)
            ymin = min(ymin, min(psd)) if ymin else min(psd)
            ymax = max(ymax, max(psd)) if ymax else max(psd)
        if not self.limits:
            yrange = ymax - ymin
            self.axes.set_ylim(
                [ymin - yrange * 0.1, ymax + yrange * 0.1]
                if yrange
                else [ymin - 1, ymax + 1]
            )
        self.fig.canvas.update()
        self.fig.canvas.flush_events()
