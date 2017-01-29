from setuptools import setup, find_packages

setup(
    name='LED Music Visualizer',
    version='0.1',
    packages=find_packages(),

    install_requires=[
        'numpy',
        'matplotlib',
        'pyserial'
    ]
)
