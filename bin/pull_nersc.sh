#!/usr/bin/env bash 

# Use function to pull the following files. Stored on NERSC
# pixel, photometry, TPS,
# pull_nersc.sh <type> <k2_camp> <run>  <<< starnames
# Example useage
# cat C2_diag120.txt | tr '\n' ' ' | pull_nersc.sh pixel C2 C2 

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
	echo ${STARS}
	FILES=$(echo ${STARS} | tr " " "\n" | awk -v dir="${REMOTEDIR}/${RUN}/output/" '{print dir $1 }' | sed '/^$/d'  | tr "\n" " "  )
	eval `echo rsync -avh --progress dtn01:"'${FILES}'"  ${OUTPUT}/`
	;;
esac

