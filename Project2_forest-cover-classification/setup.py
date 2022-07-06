import os
import sys

from setuptools import setup, find_packages

def get_requirements(requirements_path='requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]

base_dir = os.path.dirname(__file__)
src_dir = os.path.join(base_dir, 'src')
sys.path.insert(0, src_dir)

import forest_cover_classification_library



setup(
    name='',
    version='0.0.1',
    description='Library for Linear Classification Project',
    author='Philine Meyjohann',
    packages=find_packages(where='src', exclude=['tests']),
    package_dir={'': 'src'},
    install_requires=get_requirements(),
    setup_requires=['pytest-runner', 'wheel'],
    url='',
    classifiers=[
        'Programming Language :: Python :: 3.9.6'
    ])
