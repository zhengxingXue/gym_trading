from setuptools import setup

setup(name='gym_trading',
      version='0.0.0',
      install_requires=['gym',
                        'numpy',
                        'pytest',
                        'matplotlib',
                        'stable-baselines[mpi]==2.10.0',
                        'tensorflow==1.15.2',
                        'pandas']
)
