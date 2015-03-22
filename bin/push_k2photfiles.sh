#!/usr/bin/env bash
set -x
rsync -avhz --progress ${K2PHOT_DIR}/k2photfiles dtn01:${NERSCPROJ}/www/
