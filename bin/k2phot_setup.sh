# Shell script to facilitate creation of photometric jobs for k2
#!/usr/bin/env bash

echo ""
echo "Setting up batch scripts"
echo ""

STARNAME_LIST=$1 # Name of the parameter database
#TPSDIR=${HOME}/TPS/C0
#PHOTDIR=${K2_DIR}/photometry/C0/
#K2_SCRIPTS=${K2_DIR}/code/py/scripts

read -ep "Enter pixel file directory" PIXEL_DIR

# Generate a list of starnames
find ${PIXEL_DIR} -name "*.fits" |
awk -F "/" '{print $(NF)}' | # Grab the file basename
awk -F "." '{print $1}' | # Hack off the .fits part
sort > pixelfiles.temp # sorting is necessary for join

read -ep "Enter output directory " PHOTDIR
SCRIPTSDIR=${PHOTDIR}/scripts
OUTPUTDIR=${PHOTDIR}/output

mkdir -p ${PHOTDIR}
mkdir -p ${SCRIPTSDIR}
mkdir -p ${OUTPUTDIR}

# Generate a list of the stars we wish to analyze
cat ${STARNAME_LIST} | sort > starname_list.temp

# Figure out how many of the requested stars have extant photometry
join pixelfiles.temp starname_list.temp > pixel_starname_join.temp
N_PHOT_EXTANT=$(cat pixel_starname_join.temp | wc -l)
N_PARS=$(cat starname_list.temp  | wc -l )

echo "PIXEL exist ${N_PARS} out of ${N_PHOT_EXTANT} stars"
for starname in `cat pixel_starname_join.temp`
do
    PIXEL_FILE=$(grep ${STARNAME} pixelfiles.temp)
    echo "# K2_PHOT #"
    echo ". $HOME/k2_setup.sh"
    echo "cd $K2_DIR"
    echo "python ${K2PHOT_DIR}/code/py/pixel_decorrelation.py -f ${PIXEL_FILE} --wcs=1 -r 4 --minrad=2 --maxrad=8 --verbose=1 --gausscen=0 --plotmode=gs --tmin=1940 --output=pobj,fits --xymeth=xcorr2D --decor=2D --gentrend=-1.5 -s ${OUTPUTDIR}"
    echo "chmod o+rX ${OUTPUTDIR}/${STARNAME}*"
done > ${SCRIPTSDIR}/k2phot.tot
rm pixelfiles.temp starname_list.temp pixel_starname_join.temp 
