#!/bin/bash
dir=$1
echo "dir: "$dir
n=$2
echo "name: $n"
total=0
files=$(find $dir -name "$n")
for file in $files
do
	count=$(cat $file | wc -l)
	total=$((total + count))
done
echo "Total number of lines is: "$total