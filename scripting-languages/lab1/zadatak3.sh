#!/bin/bash
dir=$1
if [[ ! -d $dir ]] 
then
	echo "The given folder doesn't exist."
else
	f=$(ls $dir | egrep '^localhost_access_log.[0-9]{4}-02-[0-9]{2}.txt$')
	l=--------------------------------------------------
	for file in $f
	do
		substring1=$(echo $file | cut -d '.' -f 2)
		day=$(echo $substring1 | cut -d '-' -f 3)
		month=$(echo $substring1 | cut -d '-' -f 2)
		year=$(echo $substring1 | cut -d '-' -f 1)
		echo "datum: "$day"-"$month"-"$year
		echo $l
		cat $file | cut -d '"' -f 2 | sort | uniq -d -c | sort -r -n 
		#| sed -r "s/^([[:blank:]]*)([0-9]*)(.*)/\1\2 :\3/"
	done
fi