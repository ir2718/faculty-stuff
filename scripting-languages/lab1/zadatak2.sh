#!/bin/bash
grep -i -E 'banana|jabuka|jagoda|dinja|lubenica' ./namirnice.txt

echo
grep -i -v -E 'banana|jabuka|jagoda|dinja|lubenica' ./namirnice.txt

echo
grep -r -E '[A-Z]{3}[0-9]{6}' ~/projekti/

echo
find . -type f -mtime +7 -mtime -14 -ls 

echo
for i in {1..15}; do echo $i; done
seq 1 15

echo
kraj=15
seq 1 $kraj