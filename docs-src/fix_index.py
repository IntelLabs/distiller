#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os, fnmatch, sys


def findReplace(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)

if __name__=="__main__":
    findReplace('./site/', '/"', '/index.html"', "*.html")
    findReplace('./site/', '"/index.html"', '"./index.html"', "*.html")
    findReplace('./site/', '"."', '"./index.html"', "*.html")
    findReplace('./site/', '".."', '"../index.html"', "*.html")
    findReplace('./site/', '/"', '/index.html"', "search_index.json")
    findReplace('./site/', '/#', '/index.html#', "search_index.json")
    findReplace('./site/assets/javascripts/', 'search_index.json', 'search_index.txt', "*.js")
    findReplace('./site/mkdocs/js/', 'search_index.json', 'search_index.txt', "search.js")
    os.rename("./site/mkdocs/search_index.json", "./site/mkdocs/search_index.txt")
