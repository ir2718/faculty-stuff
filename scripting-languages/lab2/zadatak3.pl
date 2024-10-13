#!usr/bin/perl

if (@ARGV == 0) {
	while(<>) {
		chomp;
		push @lines, $_;
	}
	&line_count(@lines);
} else {
	@files = @ARGV;
	foreach $file (@files) {
		open LOG, "<", "$file";
		if(! LOG) { die "File can't be opened\n"; }

		$_ = $file;
		s/[^\.]+\.([\d]+-[\d]+-[\d]+).*/$1/;
		printf ("Datum: %s\n", $1);
		while(<LOG>) {
			push @lines, $_;
		}
		&line_count(@lines);
		
		@lines = ();
	}
}

sub line_count {
	my (@lines) = @_;
	printf ("sat: broj pristupa\n");
	printf ("-------------------------------\n");
	
	foreach (@lines){
		s/[^\:]:([\d]+).*/$1/;
		if (! exists $map{$1}) {
			$map{$1} = 1;
		} else {
			$map{$1} += 1;
		}
	}
	
	foreach $hour (sort keys %map) {
		print "  $hour : $map{$hour} \n";
	}
	
	print "\n";
	%map = ();
}