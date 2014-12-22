# Shell script to facilitate creation of photometric jobs for k2
#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Signature: k2phot_setup.sh starlist"
    exit 1
fi

echo ""
echo "Setting up batch scripts"
echo ""

STARNAME_LIST=$1 # Name of the parameter database

read -ep "Enter pixel file directory (relative path OK): " PIXEL_DIR

# Generate a list of starnames
find ${PWD}/${PIXEL_DIR} -name "*.fits" > pixelfiles.temp

cat pixelfiles.temp |
awk -F "/" '{print $(NF)}' |
awk -F "-" '{print $1}' | # Hack off the .fits part
awk -F "ktwo" '{print $2}' | # Hack off the ktwo part
sort > starnames_with_pixelfiles.temp # sorting is necessary for join

read -ep "Enter output directory " PHOTDIR
SCRIPTSDIR=${PHOTDIR}/scripts
OUTPUTDIR=${PHOTDIR}/output

mkdir -p ${PHOTDIR}
mkdir -p ${SCRIPTSDIR}
mkdir -p ${OUTPUTDIR}

# Generate a list of the stars we wish to analyze
cat ${STARNAME_LIST} | sort > starname_list.temp

# Figure out how many of the requested stars have extant pixel files
join starnames_with_pixelfiles.temp starname_list.temp > pixel_starname_join.temp

N_PIX_EXTANT=$(cat pixel_starname_join.temp | wc -l)
N_STARS=$(cat starname_list.temp  | wc -l )

echo "Pixel files exist for ${N_PIX_EXTANT}/${N_STARS} stars requested"
for STARNAME in `cat pixel_starname_join.temp`
do
    PIXEL_FILE=$(grep ${STARNAME} pixelfiles.temp)
#    PIXEL_FILE2=${OUTPUTDIR}/${STARNAME}.ffm.bin-med-std.fits
    echo "# K2_PHOT #"
    echo ". $HOME/k2_setup.sh"
    echo "cd $K2_DIR"

#    echo "python ${K2PHOT_DIR}/code/py/flatfield.py ${PIXEL_FILE} ${OUTPUTDIR} --tmin=1940"
#    echo "python ${K2PHOT_DIR}/code/py/pixel_decorrelation.py -f ${PIXEL_FILE2} --wcs=1 -r 4 --minrad=3 --maxrad=7 --verbose=1 --gausscen=0 --plotmode=gs --tmin=1940 --xymeth=xcorr2D --gentrend=-1.5 --output=pobj,fits -s ${OUTPUTDIR}"

    echo "python ${K2PHOT_DIR}/code/py/flatfield.py --tmin 1940 --tmax 1972 ${PIXEL_FILE} ${OUTPUTDIR} junk.db"
    echo "python ${K2PHOT_DIR}/code/py/pixel_decorrelation2.py --h5file ${OUTPUTDIR}/${STARNAME}.ff.h5 ${OUTPUTDIR}"
    echo "chmod o+rX ${OUTPUTDIR}/${STARNAME}*"
done > ${SCRIPTSDIR}/k2phot.tot
#rm *.temp
