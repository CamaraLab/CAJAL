#!/bin/bash

wget https://github.com/jgm/pandoc/releases/download/3.1.5/pandoc-3.1.5-linux-amd64.tar.gz
mkdir /home/docs/pandoc
tar -xvzf pandoc-3.1.5-linux-amd64.tar.gz --strip-components 1 -C /home/docs/pandoc
export PATH="/home/docs/pandoc/bin:$PATH"
echo $PATH
echo $(whereis pandoc)
echo $(which pandoc)

2023-07-12 18:33:43 (59.2 MB/s) - ‘pandoc-3.1.5-linux-amd64.tar.gz’ saved [31034735/31034735]

tar: /home/docs/pandoc: Cannot open: No such file or directory
tar: Error is not recoverable: exiting now
/home/docs/pandoc/bin:/home/docs/checkouts/readthedocs.org/user_builds/cajal/envs/optimize/bin:/home/docs/.asdf/shims:/home/docs/.asdf/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
pandoc: /usr/bin/pandoc /usr/share/pandoc
/usr/bin/pandoc
