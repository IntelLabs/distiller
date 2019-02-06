from setuptools import setup, find_packages


packages = find_packages(
     include=['distiller','distiller.*'],
     exclude=['*.__pycache__.*']
    )

with open('distiller/requirements.txt','r') as req_file:
    install_reqs = [line.strip() for line in req_file.readlines()]

setup(name='distiller',
      packages=packages,
      install_requires=install_reqs
      )

