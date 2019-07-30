## Spinningup installation

Spinup require that we use exactly Python 3.6 so if you are not using this Python version see the instructions here:
    http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/
```
    $ sudo update-alternatives --config python3
```

For Python 3.6 you may also need to install a new virtual-env:
```
    $ sudo apt-get install python3.6-venv
```

Then create and activate your venv, and populate it with the Distiller packages:
```
    $ python3 -m venv  distiller_env_python3.6
    $ source distiller_env_python3.6/bin/activate
    $ pip3 install -r requirements.txt
```

Finally, you need to install Spinup into this venv.  First clone Spinup and then install it into your venv:
```
    $ cd <spinningup-repo>
    $ sudo apt-get install python3.6-dev
    $ pip3 install -e .
```

See also:
https://spinningup.openai.com/en/latest/user/installation.html?highlight=license
