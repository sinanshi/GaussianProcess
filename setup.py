#!/usr/bin/env python

from distutils.core import setup

setup(name='gp_emulator',
      version='0.0.1',
      description='A Python GaussianProcess emulator software package',
      classifiers=[
	'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Environment :: Console'],
      author='Sinan Shi',
      author_email='sinan.shi@stats.ox.ac.uk',
      url='http://github.com/sinanshi/GaussianProcess',
      packages=['gp_emulator'],
)
