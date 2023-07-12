#!/bin/bash

wget https://github.com/jgm/pandoc/releases/download/3.1.5/pandoc-3.1.5-linux-amd64.tar.gz
tar -xvzf pandoc-3.1.5-linux-amd64.tar.gz --strip-components 1 -C /usr/local
export PATH="/usr/local:$PATH"
whereis pandoc
which pandoc
