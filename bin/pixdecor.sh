#!/usr/bin/env bash

. ${HOME}/k2_setup_erik_laptop_dev.sh


DEBUG=
DBFILE=${SM_PROJDIR}/sm_results.db
while getopts "h?:dc:r:s:t:" OPTION; do
    case $OPTION in
	d)
	    DEBUG="--debug"
	    ;;
	c)
	    K2_CAMP=${OPTARG}
	    ;;
	r)
	    RUN=${OPTARG}	    
	    ;;
	s)
	    STARNAME=${OPTARG}
	    ;;
	t)
	    TRANSFILE=${OPTARG}
	    ;;

	h|\?)
	    echo "specmatch_pipeline.sh obs [-d]" >&2
	    exit 0
	    ;;
	:)
	    echo "specmatch_pipeline.sh obs [-d]" >&2
	    exit 0
	    ;;
    esac
done

umask u=rwx,g=rx,o=rx

OUTPUTDIR=${K2_ARCHIVE}/photometry/${RUN}/output/${STARNAME}/
mkdir -p ${OUTPUTDIR}

FITSFILE=$(echo ${STARNAME} | starname_to_pixfile.sh ${K2_CAMP})
FITSFILE=${PROJDIR}/www/K2/pixel/${K2_CAMP}/${FITSFILE}
PARDB=${K2PHOTFILES}/${K2_CAMP}_pars.sqlite
RESULTSDB=${RUN_TPS_BASEDIR}/scrape.db
RUN_PHOT_BASEDIR=${K2_ARCHIVE}/photometry/${RUN}/
RUN_TPS_BASEDIR=${K2_ARCHIVE}/TPS/${RUN}/

if [ -z "${DEBUG}" ]
then
    DB=scrape_debug.db
else
    DB=scrape.db
fi

RESULTSDB=${RUN_TPS_BASEDIR}/${DB}

STAR_PHOT_OUTDIR=${RUN_PHOT_BASEDIR}/output/${STARNAME}/
STAR_TPS_OUTDIR=${RUN_TPS_BASEDIR}/output/${STARNAME}/
STAR_GRIDFILE=${STAR_TPS_OUTDIR}/${STARNAME}.grid.h5

mkdir -p ${STAR_PHOT_OUTDIR}
mkdir -p ${STAR_TPS_OUTDIR}

PIXFILE=$(echo ${STARNAME} | starname_to_pixfile.sh ${K2_CAMP})
PIXFILE=${K2_ARCHIVE}/pixel/${K2_CAMP}/${PIXFILE}
LCFILE=${STAR_PHOT_OUTDIR}/${STARNAME}.fits

echo ${PIXFILE}

PARDB=${K2PHOTFILES}/${K2_CAMP}_pars.sqlite

set -x
#pixel_decorrelation4.py ${PIXFILE} ${LCFILE} ${TRANSFILE} ${DEBUG}
#terraWrap.py pp ${LCFILE} ${STAR_GRIDFILE} ${PARDB} ${STARNAME}
#terraWrap.py grid ${STAR_GRIDFILE} ${PARDB} ${STARNAME}
#terraWrap.py dv ${STAR_GRIDFILE} ${PARDB} ${STARNAME}

echo "Saving results in ${RESULTSDB}"
scrape_terra.py ${STAR_GRIDFILE} ${RESULTSDB}
python ${K2PHOT_DIR}/code/py/lightcurve_diagnostics.py ${PIXFILE} ${LCFILE} ${RESULTSDB} ${STARNAME} --s2n=12

set +x
