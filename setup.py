from setuptools import setup, find_packages
import distiller


packages = find_packages(
     include=['distiller','distiller.*'],
     exclude=['*.__pycache__.*']
    )

with open('requirements.txt','r') as req_file:
    install_reqs = [line.strip() for line in req_file.readlines()]

setup(name='distiller',
      version=distiller.__version__,
      packages=packages,
      install_requires=install_reqs
      )

