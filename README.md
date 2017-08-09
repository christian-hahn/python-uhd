# python-uhd [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/christian-hahn/python-uhd/blob/master/LICENSE)

## Python USRP Hardware Driver Library

python-uhd is a Python C-extension that wraps the USRP hardware driver: facilitating development with USRP hardware from Python. python-uhd is Python 3 compatible. python-uhd is MIT licensed.

## Prerequisites

Install USRP Hardware Driver (UHD) software. There are several ways to achieve this. uhd-python has been tested and is compatible with the latest UHD releases including versions >= 3.8.2.

In Ubuntu, using package manager:
``` text
sudo apt-get install libuhd-dev libuhd003 uhd-host
```

Install numpy. For example, in Ubuntu using pip:
``` text
sudo pip3 install numpy
```

## Installation

Using setup.py:
``` text
git clone https://github.com/christian-hahn/python-uhd.git
cd python-uhd
python setup.py install
```

## Examples

### Receive samples

``` python
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
samps = u.receive(num_samps, channels, True)
```

## License

python-uhd is covered under the MIT licensed.

