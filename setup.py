from setuptools import setup, find_packages

import pynnmap

setup(
    name = 'pynnmap',
    version = pynnmap.__version__,
    url = 'http://github.com/lemma-osu/pynnmap/',
    author = 'LEMMA group @ Oregon State University',
    author_email = 'matt.gregory@oregonstate.edu',
    packages = find_packages(),
    description = 'Python based nearest neighbor mapping',
    install_requires=[
        'lxml',
        'matplotlib',
        'numpy',
        'pyodbc',
        'six'
    ],
)
