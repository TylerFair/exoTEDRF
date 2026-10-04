#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(name='exotedrf',
      version='2.5.0',
      license='MIT',
      author='Michael Radica',
      author_email='radicamc@uchicago.edu',
      # Include the optional JAX package.
      packages=find_packages(exclude=('tests', 'tests.*')),
      include_package_data=True,
      url='https://github.com/radicamc/exoTEDRF',
      description='Tools for Reduction of JWST TSOs',
      package_data={'': ['README.md', 'LICENSE']},
      # Install the SOSS reference files below sys.prefix/files.
      data_files=[('files', [
          'files/jwst_niriss_spectrace_0022.fits',
          'files/jwst_niriss_spectrace_0023.fits',
          'files/jwst_niriss_wavemap_0020.fits',
          'files/jwst_niriss_wavemap_0022.fits',
          'files/jwst_niriss_photom_rev2.fits',
          'files/model_background256.npy',
          'files/model_background96.npy',
      ])],
      install_requires=['applesoss==2.1.1', 'astropy', 'astroquery', 'bottleneck', 'crds',
                        'corner', 'jwst', 'matplotlib', 'more_itertools', 'numpy', 'pandas', 'ray',
                        'scikit-learn', 'scipy', 'spectres', 'requests', 'tqdm', 'pastasoss',
                        'pyyaml'],
      extras_require={'stage4': ['exotedrf', 'exouprf', 'exotic_ld', 'h5py'],
                      'webbpsf': ['exotedrf', 'webbpsf>=1.1.1'],
                      'v2-cpu': ['jax[cpu]>=0.4.30'],
                      'v2-test': ['jax[cpu]>=0.4.30', 'pytest>=7']},
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        ],
      )
