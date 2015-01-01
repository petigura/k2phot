#!/usr/bin/env bash 

# A little function that outputs the file name of pixel file given the starname and campagin number


STARNAMEFILE=$1
CAMP=$2
while read STARNAME
do
    printf 'ktwo%i-c%02d_lpd-targ.fits\n' ${STARNAME} ${CAMP:1}
done < $1

