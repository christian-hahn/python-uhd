try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
import numpy


VERSION = '2.2.0'

pyuhd = Extension(
    'pyuhd',
    define_macros=[
        ('PYUHD_VERSION', '"{}"'.format(VERSION)),
    ],
    include_dirs=[
        'uhd/',
        'uhd/auto_gen/',
        numpy.get_include(),
    ],
    libraries=['uhd'],
    extra_compile_args=['-std=c++11'],
    sources=[
        'uhd/uhd.cpp',
        'uhd/uhd_usrp.cpp',
        'uhd/uhd_types.cpp',
        'uhd/uhd_rx.cpp',
        'uhd/uhd_tx.cpp',
        'uhd/uhd_timespec.cpp',
    ],
)

setup(
    name='pyuhd',
    version=VERSION,
    description='A Python 3 C++ extension to facilitate development with USRP '
        'hardware.',
    setup_requires=['numpy'],
    install_requires=['numpy'],
    ext_modules=[pyuhd],
    author='Christian Hahn',
    author_email='christianhahn09@gmail.com',
    long_description='python-uhd is a Python 3 C++ extension to facilitate '
        'development with USRP hardware from Python.',
    url='https://github.com/christian-hahn/python-uhd',
    license='MIT',
)
