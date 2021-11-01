import astropy.io.fits as fits
import drizzlepac
import os

import grizli
import grizli.utils
print('===> fetching default calibretion data')
grizli.utils.fetch_default_calibs()
print('===> fetching configs')
grizli.utils.fetch_config_files()
from grizli.pipeline import auto_script
from grizli import utils, fitting, multifit, prep
from reprocess_wfc3 import reprocess_wfc3

# force remake symlinks. For other installation procedures, see: https://grizli.readthedocs.io/en/master/grizli/install.html
utils.symlink_templates(force=True)
utils.set_warnings()

print('grizli package:', grizli)
print('reprocess_wfc3 package:', reprocess_wfc3)
