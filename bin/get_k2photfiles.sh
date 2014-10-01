#!/usr/bin/env bash
OLDPWD=${PWD}
K2PHOTFILES=${PWD}/k2photfiles
CATDIR=${K2PHOTFILES}/catalogs/
REMCATDIR=http://portal.nersc.gov/project/m1669/k2photfiles/catalogs/
echo "Downloading k2photfiles into ${CATDIR}"
mkdir -p ${CATDIR}

wget ${REMCATDIR}/k2_catalogs.h5 -O ${CATDIR}/k2_catalogs.h5 
wget ${REMCATDIR}/k2_catalogs.sqlite -O ${CATDIR}/k2_catalogs.sqlite

echo "### Download finished ####"
echo "Add to k2_setup.sh (or .bashrc)"
echo ""
echo "export K2PHOTFILES=${K2PHOTFILES}"

