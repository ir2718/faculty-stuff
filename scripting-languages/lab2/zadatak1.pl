#!usr/bin/perl

print "Enter some text:\n";
$text = <STDIN>;

print "Enter a positive whole number:\n";
$n = <STDIN>;

foreach (1..$n) {
	print "$text";
}	
