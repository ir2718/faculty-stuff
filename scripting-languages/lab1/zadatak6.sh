#!/bin/bash
dir1=$1
dir2=$2
if [[ ! -d $dir1 ]]
then
	echo "The first directory doesn't exist."
elif [[ ! -d $dir2 ]]
then
	echo "The second directory doesn't exist."
else
	files1=$(find $dir1 -type f)
	files2=$(find $dir2 -type f)	
	for file1 in $files1
	do
		found=0
		for file2 in $files2
		do
			if [ ${file1##*/} = ${file2##*/} ]
			then
				found=1
				if [ $file1 -nt $file2 ]
				then
					echo $file1" --> "$dir2
				fi
			fi
		done
		
		if [ $found -eq 0 ]
		then
			echo $file1" --> "$dir2
		fi
	done
	
	for file2 in $files2
	do
		found=0
		for file1 in $files1
		do
			if [ ${file1##*/} = ${file2##*/} ]
			then
				found=1
				if [ $file2 -nt $file1 ]
				then
					echo $file2" --> "$dir1
				fi
			fi
		done
		
		if [ $found -eq 0 ]
		then
			echo $file2" --> "$dir1
		fi
	done
fi
