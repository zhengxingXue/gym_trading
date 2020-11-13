from setuptools import setup

setup(name='gym_trading',
      version='1.0.0',
      install_requires=['gym',
                        'numpy',
                        'pytest',
                        'matplotlib',
                        'stable-baselines[mpi]==2.10.0',
                        'tensorflow==2.3.1',
                        'pandas']
)
