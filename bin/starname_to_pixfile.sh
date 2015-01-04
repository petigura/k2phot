#!/usr/bin/env bash 

# A little function that outputs the file name of pixel file given the starname and campagin number

STARS=$(cat)
CAMP=$1

for STAR in $STARS
do
    printf 'ktwo%i-c%02d_lpd-targ.fits\n' ${STAR} ${CAMP:1}
done 

