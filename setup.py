try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from os.path import join
import numpy


MAJOR_VERSION = 0
MINOR_VERSION = 1

uhd = Extension(
    'uhd',
    define_macros=[
        ('MAJOR_VERSION', str(MAJOR_VERSION)),
        ('MINOR_VERSION', str(MINOR_VERSION)),
    ],
    include_dirs=[numpy.get_include(), 'uhd/'],
    libraries=['uhd'],
    extra_compile_args=['-std=c++11'],
    sources=[
        'uhd/uhd.cpp',
        'uhd/uhd_object.cpp',
        'uhd/uhd_types.cpp',
        'uhd/uhd_rx.cpp',
        'uhd/uhd_tx.cpp',
        'uhd/uhd_timespec.cpp',
    ],
)

setup(
    name='uhd',
    version='{}.{}'.format(MAJOR_VERSION,MINOR_VERSION),
    description='A Python 3 C-extension to facilitate development with USRP '
        'hardware.',
    setup_requires=['numpy'],
    install_requires=['numpy'],
    ext_modules=[uhd],
    author='Christian Hahn',
    author_email='christianhahn09@gmail.com',
    packages=['uhd'],
    long_description='python-uhd is a Python 3 C-extension to facilitate '
        'development with USRP hardware from Python.',
    license='MIT',
)
