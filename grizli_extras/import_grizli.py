import astropy.io.fits as fits
import drizzlepac
import os

import grizli
import grizli.utils
# grizli.utils.fetch_default_calibs()
from grizli.pipeline import auto_script
from grizli import utils, fitting, multifit, prep
from reprocess_wfc3 import reprocess_wfc3
from grizli.ds9 import DS9

# force remake symlinks. For other installation procedures, see: https://grizli.readthedocs.io/en/master/grizli/install.html
utils.symlink_templates(force=True)
utils.set_warnings()

print('grizli package:', grizli)
print('reprocess_wfc3 package:', reprocess_wfc3)

from mastquery import query, overlaps
from grizli.pipeline.auto_script import get_yml_parameters
