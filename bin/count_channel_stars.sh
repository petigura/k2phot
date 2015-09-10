#!/usr/bin/env bash

HEADERSDB=$1

while getopts "h?" OPTION; do
    case ${OPTION} in
	h)
	    echo "Count stars in header database"
	    echo "Example useage"
	    echo "--------------"
	    echo "count_channel_stars.sh C3_headers.db"
	    echo "channel stars_in_channel"
	    echo "1 142"
	    echo "2 191"
	    echo "3 280"
	    echo "<snip>"
	    echo "82 316"
	    echo "83 281"
	    echo "84 221"
	    exit 0
	    ;;
    esac
done


echo "mod   out   channel  stars_in_channel"
echo "-------------------------------------"
sqlite3 $HEADERSDB <<EOF
.separator "\t"
SELECT module,output,channel,count(*) 
FROM headers 
GROUP BY
channel;
EOF
echo "------------------------------------"
echo "The following channels are on the outer modules" 
echo "1-16"
echo "29-32"
echo "33-36"
echo "49-52"
echo "53-56"
echo "69-84"
