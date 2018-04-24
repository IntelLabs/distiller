How to generate the Distiller documentation site
================================================
1. Install mkdocs and the required packages by executing:

```
$ pip3 install -r doc-requirements.txt
```

2. To build the project documentation run:
```
$ cd distiller/docs-src
$ mkdocs build --clean
```
This will create a folder named 'site' which contains the documentation website.
Open distiller/docs/site/index.html to view the documentation home page.
