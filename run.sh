#!/bin/bash

declare -a set=('ch1' 'ch2' 'tr1' 'tr2')
k=0

#for Dir in $HOME/Bioimage/Images/*/*/ #server
for Dir in $HOME/workspace/Bioimage2/Images/*/*/subset/ #h laptop
do
   #echo $Dir ${set[k]}
   python3 cell_count.py $Dir ${set[k]}
   k=$(($k + 1))
done
