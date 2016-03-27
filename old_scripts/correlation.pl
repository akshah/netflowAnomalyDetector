use Statistics::Basic qw(:all);
use Data::Dumper;

my $file = $ARGV[0];
open(FILE,'<',$file) or die "can't open $file: $!";
my @data;
my @raw_data;
while (<FILE>) {
	my @row = split ',', $_;
	push @raw_data, \@row;
}
#print "Number of Columns is $num_cols\n";
for my $row (@raw_data) {
	for my $column (0 .. $#{$row}) {
		push(@{$data[$column]}, $row->[$column]);
	}
}
#print Dumper(@data);

my $num_rows=@data;
for (my $j=0; $j<$num_rows;$j++) {
	for (my $i=0; $i<$num_rows;$i++) {
		my $correlation = correlation($data[$j],$data[$i]);
		print "$correlation";
		if($i!=$num_rows-1){
			print ",";
		}
	}
	print "\n";
}

=begin
while(<FILE>){

#my $mean = mean(1,2,3);

}
print "Mean is $mean\n";
=cut
close(FILE);
