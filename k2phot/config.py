import os

bjd0 = 2454833

required_environment_variables = [
    'K2_ARCHIVE','K2PHOTFILES','K2PHOT_DIR','K2_DIR']

for var in required_environment_variables:
    assert os.environ.has_key(var), "%s environment variable must be set"  % var
    assert os.environ[var]!='',"%s cannot be empty string" % var 
    exec('%s = os.environ["%s"]' % (var,var) )


