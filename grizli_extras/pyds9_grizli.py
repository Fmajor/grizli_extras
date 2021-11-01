import pyds9
class DS9(pyds9.DS9):
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    #<<170417>>XIN: copied from Gabe Brammer's pyds9
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def view(self, img, header=None):
        """
        From pysao
        """
        if hasattr(img, 'header'):
            ### FITS HDU
            self.set_np2arr(img.data)
            if not header:
                header = img.header
            self.set("wcs replace", get_wcs_headers(header))
        else:
            self.set_np2arr(img)
            if header:
                self.set("wcs replace", get_wcs_headers(header))

    def frame(self, id):
        self.set('frame %d' %(id))

    def scale(self, z1, z2):
        self.set('scale limits %f %f' %(z1, z2))

    def set_defaults(self, match='image', verbose=False):
        """
        Match frame, set log scale
        """
        commands = """
        xpaset -p ds9 scale log
        xpaset -p ds9 scale limits -0.1 10
        xpaset -p ds9 cmap value 3.02222 0.647552
        xpaset -p ds9 match frames %s
        xpaset -p ds9 frame lock %s
        xpaset -p ds9 match colorbars
        xpaset -p ds9 lock colorbar
        xpaset -p ds9 match scales""" %(match, match)

        for c in commands.split('\n'):
            if 'xpaset' in c:
                self.set(' '.join(c.split()[3:]))
                if verbose:
                    print (c)

    def match(self, match='image'):
        commands = """
        xpaset -p ds9 match frames %s
        xpaset -p ds9 frame lock %s
        """ %(match, match)

        for c in commands.split('\n'):
            if 'xpaset' in c:
                self.set(' '.join(c.split()[3:]))

def get_wcs_headers(h):
    """
    given a fits header, select only wcs related items and return them
    in a single string
    GBrammer - taken from pysao
    """
    import re

    wcs_key_pattern = re.compile(r'^(NAXIS|CD|CDELT|CRPIX|CRVAL|CTYPE|CROTA|LONGPOLE|LATPOLE|PV|DISTORT|OBJECT|BUNIT|EPOCH|EQUINOX|LTV|LTM|DTV|DTM|PC|CUNIT|RADESYS|WCSNAME|A_|B_)')

    try:
        cardlist = h.ascardlist()
    except:
        ### newer astropy.io.fits
        cardlist = h.cards

    l =[s.image for s in cardlist if wcs_key_pattern.match(s.keyword)]

    return "\n".join(l)
