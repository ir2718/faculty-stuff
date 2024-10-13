#!usr/bin/perl

if(@ARGV != 1){
	die "The script takes in exactly 1 argument; path_to_file";
}

$row = 0;
open FILE, "<", "$ARGV[0]";

foreach (<FILE>) {
	if(m/^#/) { 
	
	} elsif ($row == 0){
		@coefs =  split /;/, $_;
		$row = 1;
	} else {
		@info = split /;/, $_;
		
		$index = 0;
		$v = 0.0;
		foreach $c (@coefs) {
			if($info[3+$index] ne "-") {
				$v += $c * $info[3+$index];
				$index += 1;
			}
		}
		
		$k = "$info[1], $info[2] ($info[0])";
		$map{$k} = $v;
	}
}

print "Lista po rangu:\n";
print "-------------------\n";

$counter = 0;
foreach $key (reverse sort { $map{$a} <=> $map{$b} } keys %map) {
	printf "%4d. %-35s : %.2f\n", $counter, $key, $map{$key};
	$counter += 1;
}