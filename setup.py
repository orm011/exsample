from setuptools import find_packages
from distutils.core import setup
setup(
    name = 'exsample',
    version = '1.0.0',
    url = 'https://github.com/mypackage.git',
    author = 'Oscar Moll',
    author_email = 'orm@mit.edu',
    description = 'Description of my package',
    packages = find_packages('exsample'),
    install_requires = [],
)
