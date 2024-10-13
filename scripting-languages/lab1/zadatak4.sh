#!/bin/bash
dir=$1
if [[ ! -d $dir ]] 
then
	echo "The given folder doesn't exist."
else
	dates=$(ls $dir | egrep '^[0-9]{8}_[0-9]{4}.jpg$' | cut -c1-6 | sort | uniq)
	for date in $dates
	do
		m="${date:4:2}"
		y="${date:0:4}"
		echo $m"-"$y" :"
		echo "----------"
		# echo $(ls -1 $dir | egrep "^$date" | sort) 
		ls -1 $dir | egrep "^$date" | sort | cat --number | sed -r "s/(^[[:blank:]]*[0-9]*)([[:blank:]]*)/\1.\2/"
		echo "--- Ukupno: "$(ls -1 $dir | egrep "^$date" | sort | wc -l)" slika -----"
		echo
	done
fi