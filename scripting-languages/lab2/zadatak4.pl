#!usr/bin/perl

if(@ARGV == 0){
	while(<>) {
		chomp;
		push @lines, $_;
	}
	&find_late_students(@lines);
} elsif (@ARGV == 1) {
	open FILE, "<", "$ARGV[0]";
	if(! FILE) { die "File can't be opened\n"; }
	
	foreach (<FILE>) {
		chomp $_;
		push @lines, $_;
	}
	&find_late_students(@lines);
}

sub find_late_students {
	my(@lines) = @_;
	@first = split /;/, (shift @lines);
	
	$i = 0;
	foreach (@first) {
		$map{$_} = $i;
		$i += 1;
	}
	
	foreach $line (@lines) {
		@info = split /;/, $line;
		
		$_ = $info[$map{"Termin"}];
		s/([0-9]{4}-[0-9]{2}-[0-9]{2}) ([0-9]{2}).*/$1 $2/;
		$old_h = $2;
		$old_d = $1;
		
		$_ = $info[$map{"Zakljucano"}];
		s/([0-9]{4}-[0-9]{2}-[0-9]{2}) ([0-9]{2}).*/$1 $2/;
		$new_h = $2;
		$new_d = $1;
		
		if($new_d ne $old_d or $new_h > $old_h) {
			@info_fields = split / /, $info[$map{"Termin"}];
			$datetime1 = $info_fields[0]." ".$info_fields[1];
			printf ("%s %s %s - PROBLEM: %s --> %s\n", 
				$info[$map{"JMBAG"}], $info[$map{"Prezime"}], $info[$map{"Ime"}], $datetime1, $info[$map{"Zakljucano"}]);
		}
	}
	
	undef %map;
}