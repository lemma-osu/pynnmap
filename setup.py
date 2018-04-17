from setuptools import setup, find_packages

import pynnmap

setup(
    name='pynnmap',
    version=pynnmap.__version__,
    url='http://github.com/lemma-osu/pynnmap/',
    author='LEMMA group @ Oregon State University',
    author_email='matt.gregory@oregonstate.edu',
    packages=find_packages(),
    description='Python based nearest neighbor mapping',
    install_requires=[
        'click',
        # 'impute',
        'lxml',
        'matplotlib',
        'numpy',
        'pandas',
        'rpy2',
        'scipy',
        'six'
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            'tox'
        ]
    },
    entry_points='''
        [console_scripts]
        pynnmap=pynnmap.cli.main:main_group

        [pynnmap.cli_commands]
        build-model=pynnmap.cli.build_model:build_model
        cross-validate=pynnmap.cli.cross_validate:cross_validate
        find-outliers=pynnmap.cli.find_outliers:find_outliers
        impute=pynnmap.cli.impute:impute
        new-targets=pynnmap.cli.new_targets:new_targets
    ''',
)
