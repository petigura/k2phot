#!/usr/bin/env bash 

# Use function to pull the following files. Stored on NERSC
# pixel, photometry, TPS,
# pull_nersc.sh <type> <k2_camp> <run>  <<< starnames

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
    photometry*)
	OUTPUT=${LOCALDIR}/${RUN}/output/
	mkdir -p ${OUTPUT}
	echo ${STARS}
	FILES=$(echo ${STARS} | tr " " "\n" | awk -v dir="${REMOTEDIR}/${RUN}/output/" '{print dir $1 }' | sed '/^$/d'  | tr "\n" " "  )
#	echo ${FILES}

	echo rsync -avh --progress dtn01:"'${FILES}'"  ${OUTPUT}/


#	rsync -avh --progress dtn01:${REMOTEDIR}/${RUN}/scripts/ ${LOCALDIR}/${RUN}/
#	rsync -vh --progress dtn01:${REMOTEDIR}/${RUN}/scrape.db ${LOCALDIR}/${RUN}/
	;;

    TPS*)
	;;
esac




#echo ${FILES}
