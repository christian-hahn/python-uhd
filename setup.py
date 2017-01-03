try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import numpy

MAJOR_VERSION = 0
MINOR_VERSION = 1

uhd = Extension('uhd',
                define_macros = [('MAJOR_VERSION', str(MAJOR_VERSION)),
                                 ('MINOR_VERSION', str(MINOR_VERSION))],
                include_dirs = [numpy.get_include(), 'uhd/'],
                libraries = ['uhd'],
                library_dirs = [],
                extra_compile_args=['-std=c++11'],
                sources = ['uhd/' + s for s in ['uhd.cpp', 'uhd_types.cpp',
                                                'uhd_gen.cpp', 'uhd_rx.cpp']])

setup(name = 'uhd',
      version = '{}.{}'.format(MAJOR_VERSION,MINOR_VERSION),
      description = 'A Python 3 C-extension to facilitate development with USRP hardware.',
      ext_modules = [uhd],
      author = 'Christian Hahn',
      author_email = 'christianhahn09@gmail.com',
      url = '',
      packages = ['uhd'],
      long_description = """python-uhd is a Python 3 C-extension to facilitate development with USRP hardware from Python.""",
      license = 'MIT')
