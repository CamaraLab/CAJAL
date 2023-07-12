#!/bin/bash

wget https://github.com/jgm/pandoc/releases/download/3.1.5/pandoc-3.1.5-linux-amd64.tar.gz
mkdir pandoc-dir
tar -xvzf pandoc-3.1.5-linux-amd64.tar.gz --strip-components 1 -C /home/docs/pandoc
export PATH="/home/docs/pandoc/bin:$PATH"
echo $PATH
echo $(whereis pandoc)
echo $(which pandoc)
