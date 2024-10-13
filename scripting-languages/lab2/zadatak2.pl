#!usr/bin/perl

@numbers = <STDIN>;

$mean = 0;
foreach $num (@numbers) {
	$mean += $num;
}
$mean /= ($#numbers + 1);

print "Calculated mean is: $mean";