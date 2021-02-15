from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='vkusmart',
    version='0.1.0',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
)

extras_require={
  'extra_scope': [
    'torch >= 1.0.0',
    'torchvision',
    'git+https://github.com/ncullen93/torchsample',
    'nibabel',
    'tqdm',
  ],
}