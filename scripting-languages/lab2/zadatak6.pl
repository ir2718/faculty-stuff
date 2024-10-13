#!usr/bin/perl
use open ':locale';
use locale;

$len = @ARGV[-1];

foreach $i (0..$#ARGV) {
	open FILE, "<", "$ARGV[$i]";

	foreach $line (<FILE>) {
		$_ = $line;
		s/[[:punct:]]//g;
		s/[\s]+/ /g;
		@words = split / /, $_;
		
		foreach $w (@words) {
			$w = lc(substr($w, 0, $len));
			if(length($w) == $len) {
				if(exists $map{$w}) {
					$map{$w} += 1;
				} else {
					$map{$w} = 1;
				}
			}
		}
	}
}

foreach(sort keys %map) {
	print "$_ : $map{$_}\n";
}