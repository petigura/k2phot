#!/usr/bin/env bash 


while getopts "h?:dc:r:s:t:e:" OPTION; do
    case $OPTION in
	h|\?)
	    echo "Use function to pull the following files. Stored on NERSC"
	    echo "pixel, photometry, TPS,"
	    echo "pull_nersc.sh <type> <k2_camp> <run>  <<< starnames"
	    echo "Example useage"
	    echo "cat C2_diag120.txt | tr '\n' ' ' | pull_nersc.sh pixel C2 C2"
	    echo "cat C2_diag120.txt | tr '\n' ' ' | pull_nersc.sh photometry C2 C2_03-21"
	    exit 0
	    ;;
    esac
done


STARS=$(cat)

TYPE=$1
CAMP=$2
RUN=$3

REMOTEDIR=${NERSCPROJ}/www/K2/${TYPE}
LOCALDIR=${K2_ARCHIVE}/${TYPE}

case ${TYPE} in
    pixel*)
	FILES=$( 
	    echo ${STARS} | starname_to_pixfile.sh ${CAMP}
	)
	rsync -avhz --progress --files-from=<( echo ${FILES} | tr " "  "\n " ) dtn01:${REMOTEDIR}/${RUN}/ ${LOCALDIR}/${RUN}/

	;;
    photometry|TPS)
	OUTPUT=${LOCALDIR}/${RUN}/output/
	mkdir -p ${OUTPUT}
	FILES=$(echo ${STARS} | tr " " "\n" | awk -v dir="${REMOTEDIR}/${RUN}/output/" '{print dir $1 }' | sed '/^$/d'  | tr "\n" " "  )
	set -x
	eval `echo rsync -avh --progress dtn01:"'${FILES}'"  ${OUTPUT}/ --exclude='*.png' --exclude='*.pdf'`
	;;
esac

