#!/bin/bash

wget https://github.com/jgm/pandoc/releases/download/3.1.5/pandoc-3.1.5-1-amd64.deb
dpkg -i pandoc-3.1.5-1-arm64.deb --force-not-root --root=/usr/local
