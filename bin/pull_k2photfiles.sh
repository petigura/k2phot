#!/usr/bin/env bash
set -x
rsync -avhz --progress dtn01:${NERSCPROJ}/www/k2photfiles ${K2PHOT_DIR}/ 
