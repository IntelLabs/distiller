import os

#from pip.download import PipSession
#from pip.req import parse_requirements
from setuptools import setup

abs_path = os.path.dirname(os.path.abspath(__file__))
requirements_file = os.path.join(abs_path, 'requirements.txt')
#install_requirements = parse_requirements(requirements_file, session=PipSession())
#requirements = [str(ir.req) for ir in install_requirements]

setup(
    name='distiller_adc',
    version='0.0.1',
    description='OpenAI Gym interface to Distiller',
    author='Intel AI Lab, Haifa',
    author_email='neta.zmora@intel.com',
    license='Apache',
    packages=['distiller_adc'],
    install_requires=['gym==0.9.4', 'numpy', 'networkx', 'matplotlib'],
)
