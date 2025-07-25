#!/usr/bin/perl
use IO::Socket::INET;

my $socket = IO::Socket::INET->new(
    PeerAddr => 'localhost:6333',
    Proto => 'tcp',
    Timeout => 2
);

if ($socket) {
    print $socket "GET / HTTP/1.0\r\n\r\n";
    while (<$socket>) {
        if (/200 OK/) {
            close($socket);
            exit 0;
        }
    }
    close($socket);
}

exit 1;