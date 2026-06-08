# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyOMA',
    version='0.1.0a1',
    description='Package for Operational Modal Analysis in Python',
    long_description=readme,
    author='Simon Marwitz',
    author_email='simon.jakob.marwitz@uni-weimar.de',
    url='https://santafe.bauing.uni-weimar.de/pyOMA',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True
)
