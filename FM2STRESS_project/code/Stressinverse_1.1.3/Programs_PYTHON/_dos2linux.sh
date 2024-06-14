#! /bin/sh

for file in `ls *.py`
do
 cat -v $file | sed 's/\^M//g' > pom.tmp
 mv pom.tmp $file
done

chmod 755 StressInverse.py
