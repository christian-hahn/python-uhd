# python-uhd [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/christian-hahn/python-uhd/blob/master/LICENSE)

## Python USRP Hardware Driver Library

python-uhd is a Python C-extension that wraps the USRP hardware driver: facilitating development with USRP hardware from Python.  python-uhd is Python 3 compatible.  python-uhd is MIT licensed.

## Prerequisites

Install USRP Hardware Driver (UHD) software.  There are several ways to achieve this.  python-uhd has been tested and is compatible with the following UHD releases:

* v3.9.0 - v3.9.7
* v3.10.0.0 - v3.10.3.0
* v3.11.0.0 - v3.11.1.0
* v3.12.0.0
* v3.13.0.0 - v3.13.1.0
* v3.14.0.0 - v3.14.1.1
* v3.15.0.0
* v4.0.0.0
* v4.1.0.0 - v4.1.0.1

In Ubuntu, using package manager:
``` text
sudo apt-get install libuhd-dev libuhd003 uhd-host
```

Install numpy.  For example, in Ubuntu using pip:
``` text
sudo pip3 install numpy
```

## Installation

Using setup.py:
``` text
git clone https://github.com/christian-hahn/python-uhd.git
cd python-uhd/
sudo python3 setup.py install
```

## Examples

### Receive samples
```python
from pyuhd import Usrp

# Create USRP object
u = Usrp()

# Parameters
center_freq = 140.625e6
sample_rate = 10.0e6
num_samples = 2**16
channels = range(u.get_rx_num_channels())

# Set sample rate
u.set_rx_rate(sample_rate)

# For each channel
for chan in channels:
    u.set_rx_bandwidth(sample_rate, chan)
    u.set_rx_freq(center_freq, chan)
    u.set_rx_antenna('RX2', chan)
    u.set_rx_gain(40., chan)
```
#### Method A
Not streaming, one-shot and done.
```python
samples = u.receive(
    num_samples,
    channels,
)
```
#### Method B
Streaming without recycle.
```python
u.receive(
    num_samples,
    channels,
    streaming=True,
)
samples = u.receive()
u.stop_receive()
```
#### Method C
Streaming with recycle.  When `recycle=True`, one block of samples is always ready
to be read.  As old blocks of samples become stale, they are discarded, and the
underlying buffers are recycled to be used for the next, newer block of samples.
```python
u.receive(
    num_samples,
    channels,
    streaming=True,
    recycle=True,
)
samples = u.receive(fresh=True)
u.stop_receive()
```
`fresh=True` guarantees that the time of the first sample returned is after the call to
`receive()`, that the samples are indeed fresh, not stale.  If this is not required,
`fresh=False` will yield samples that have been buffered.
## License
python-uhd is covered under the MIT licensed.
