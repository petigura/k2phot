#!/usr/bin/env bash
rsync -avhz --progress ${K2PHOT_DIR}/k2photfiles dtn01:${NERSCPROJ}/www/
