from importlib import reload
from functools import reduce
from termcolor import colored
import os
import sys
import glob
import yaml
import time
import random
import pickle
from pathlib import Path
import pprint
import shutil

import numpy as np
import matplotlib.pyplot as plt
import copy

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column, join, vstack, hstack, setdiff, unique
from astropy.time import Time

from IPython.display import Image
from IPython.display import FileLink, FileLinks
from IPython.core.display import display, HTML, Markdown, Latex
from IPython.display import IFrame
# from wand.image import Image as WImage
from astropy.visualization import ZScaleInterval
import astropy.io.fits as fits
from matplotlib.gridspec import GridSpec
import grizli
import grizli.pipeline
reload(grizli)
reload(grizli.pipeline)
from grizli import utils, fitting, multifit, prep
from grizli.pipeline import auto_script
from mocpy import MOC, World2ScreenMPL
from astropy.coordinates import Angle, SkyCoord

from astroquery.mast import Observations
from astroquery.simbad import Simbad

## from https://stackoverflow.com/questions/19470099/view-pdf-image-in-an-ipython-notebook
class PDF(object):
  def __init__(self, pdf, size=(600,600)):
    self.pdf = pdf
    self.size = size

  def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

def thtml(text, color='black', fontsize='20px', center=True):
  if center:
    html_ = "<div style=text-align:center><b style=color:{color};font-size:{fontsize}>{text}</b></div>".format(**locals())
  else:
    html_ = "<div><b style=color:{color};font-size:{fontsize}>{text}</b></div>".format(**locals())
  display(HTML(html_))

def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current,
        toggle_text=toggle_text
    )

    return HTML(html)

# foldinput
def autofold(dofold=False):
  from IPython.display import HTML
  from IPython.display import display as Display
  import time
  script_html = '''
  <script id="lab-custom-folding">
  function foldall(el) {
    minput  = /^#.*(foldinput)/
    moutput = /^#.*(foldoutput)/
    var cells = document.getElementsByClassName('jp-Cell')
    for (var cell of cells) {
      var input = cell.getElementsByClassName('jp-Cell-inputWrapper')[0]
      var input_line = input.getElementsByClassName('CodeMirror-line')
      if (!input_line.length) continue
      var output = cell.getElementsByClassName('jp-Cell-outputWrapper')[0]
      input_line = input_line[0].textContent
      if (!minput.test(input_line) && !moutput.test(input_line)) continue

      var input_collapser  = cell.getElementsByClassName('jp-Cell-inputCollapser')[0].children[0]
      var output_collapser = cell.getElementsByClassName('jp-Cell-outputCollapser')[0].children[0]
      var input_area  = cell.getElementsByClassName('jp-InputArea')
      var output_area = cell.getElementsByClassName('jp-OutputArea')
      // console.log(input_collapser, output_collapser, input_area, output_area)

      if (minput.test(input_line) && input_area.length>0) {
        input_collapser.click()
      }
      if (moutput.test(input_line) && output_area.length>0) {
        output_collapser.click()
      }
    }
  }
  function unfoldall(el) {
    var cells = document.getElementsByClassName('jp-Cell')
    for (var cell of cells) {
      var input = cell.getElementsByClassName('jp-Cell-inputWrapper')[0]
      var output = cell.getElementsByClassName('jp-Cell-outputWrapper')[0]

      var input_collapser  = cell.getElementsByClassName('jp-Cell-inputCollapser')[0].children[0]
      var output_collapser = cell.getElementsByClassName('jp-Cell-outputCollapser')[0]
      if (output_collapser) {
        output_collapser = output_collapser.children[0]
      }
      var input_area  = cell.getElementsByClassName('jp-InputArea')
      var output_area = cell.getElementsByClassName('jp-OutputArea')
      // console.log(input_collapser, output_collapser, input_area, output_area)

      if (input_area.length==0) {
        input_collapser.click()
      }
      if (output_collapser && output_area.length==0) {
        output_collapser.click()
      }
    }
  }
  </script>''' + '''
  <button id="autofold" onclick="foldall(document.getElementById('autofold'))">
    foldall
  </button>
  <button id="autounfold" onclick="unfoldall(document.getElementById('autounfold'))">
    unfoldall
  </button>
  '''
  if dofold:
    script_html = script_html + '''
    <script>
      document.getElementById('autofold').click()
    </script>
    '''
  Display(HTML(script_html))

def show_mast_query(table):
  toshow = table.copy();
  target_classification = table['target_classification']
  obs_title = table['obs_title']
  print('target_classification:', list(np.unique(target_classification)))
  print('obs_title', list(np.unique(obs_title)))
  toshow.remove_columns([
    'footprint', 'jpegURL', 'calib_level',
    'obs_id', 'objID', #'obsid', 
    'proposal_id', 'sequence_number',
    'dataURL', 'mtFlag', 'intentType',
    'ra', 'dec',
    't_min', 't_max',
    'em_min', 'em_max',
    'obs_collection',
    'proposal_pi',
    'target_classification',
    'obs_title',
    't_obs_release',
    ])

  return toshow

def generate_raw_drz_list():
  visits = glob.glob("j*visits.npy")
  for eachfile in visits:
    filters = set()
    pname = eachfile.split('_visits')[0]
    visits, all_groups, info = np.load(eachfile, allow_pickle=True)
    drfiles = {}
    for eachgroup in all_groups:
      for itype in ['grism', 'direct']:
        ds10_list = []
        gdata = eachgroup[itype]
        f = gdata['product'].split('-')[-1]
        if f not in drfiles:
          drfiles[f] = []
        filters.add(f)
        #product = '-'.join(grism['product'].split('-')[:-1])
        product = gdata['product']
        ds10_file = "{}.ds10.txt".format(product)
        drz_file = product+'_drz_sci.fits'
        drfiles[f].append(drz_file)
        ds10_list.extend(gdata['files'])
        ds10_list.append(drz_file)
        with open(ds10_file, 'w') as f:
          f.write('\n'.join(ds10_list))
    for fname in list(filters):
      with open('{}-{}.combined.ds10.txt'.format(pname, fname), 'w') as f:
        f.write('\n'.join(drfiles[fname] + ["{}-{}_drz_sci.fits".format(pname, fname)]))

def zscale_imshow(data,
    grids=None,
    kwargs_fig=dict(),
    kwargs_gs=dict(),
    kwargs_imshow=dict(),
    title=None,
  ):
  z = ZScaleInterval()
  flags = {'title_flag': False}
  def imshow(ax, data):
    vmin,vmax = z.get_limits(data)
    if data.dtype == np.bool:
      vmin = 0
      vmax = 1
    extent = [0, data.shape[1], 0, data.shape[0]]
    kwargs = dict(
        origin='lower',
        interpolation='Nearest',
        cmap=plt.cm.viridis,
        aspect='equal',
        extent=extent,
    )
    kwargs.update(kwargs_imshow)
    ax.imshow(data,  vmin=vmin, vmax=vmax, **kwargs)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    if not flags['title_flag'] and title is not None:
      ax.set_title(title)
      flags['title_flag'] = True
  
  fig = plt.figure(constrained_layout=True, **kwargs_fig)
  if grids is not None: # data should be 1D array, grids should be like {'s': [a,b], 'l':[sliceA, sliceB, ...]}
    gs = fig.add_gridspec(*grids['s'], **kwargs_gs)
    for data,s in zip(data, grids['l']):
      ax = fig.add_subplot(gs[s])
      imshow(ax, data)
  else: # build gs row by row with data list's shape
    if not isinstance(data, list):
      ax = fig.add_subplot(111)
      imshow(ax, data)
    else:
      nrows = len(data)
      ncolumns = 0
      for row in data:
        if len(row) > ncolumns:
          ncolumns = len(row)
      gs = fig.add_gridspec(nrows, ncolumns, **kwargs_gs, wspace=0, hspace=0)
      for irow,drow in enumerate(data):
        for icol,dcol in enumerate(drow):
          if dcol is not None:
            ax = fig.add_subplot(gs[irow, icol])
            imshow(ax, dcol)
  return fig

def test():
  print('kuku')

def show_flt_images(flt, figsize=(15,7.5)):
  z = ZScaleInterval()
  filter = flt.grism.filter
  f = flt.grism_file
  PA = flt.dispersion_PA
  def imshow(ax, data):
    vmin,vmax = z.get_limits(data)
    extent = [0, data.shape[0], 0, data.shape[1]]
    kwargs = dict(
        origin='lower',
        interpolation='Nearest',
        cmap=plt.cm.viridis,
        aspect=data.shape[1]/data.shape[0],
    )
    ax.imshow(data,  vmin=vmin, vmax=vmax, extent=extent, **kwargs)
    ax.set(xticklabels=[], yticklabels=[])
    ax.xaxis.set_tick_params(length=0); ax.yaxis.set_tick_params(length=0)
  model  = flt.model  
  fig = plt.figure(constrained_layout=True, figsize=figsize)
  gax = fig.add_subplot(111)
  gax.set(xticks=[], yticks=[])
  gax.set_title('{f} filter:{filter} PA:{PA}'.format(**locals()))
  gax.set_aspect(model.shape[1]/model.shape[0])
  
  gs = plt.GridSpec(1, 2, wspace=0, hspace=0)
  grism    = flt.direct.data['SCI']
  direct   = flt.direct.data[flt.direct.thumb_extension]
  ax = fig.add_subplot(gs[0,0])
  imshow(ax, direct); ax.set_anchor('E')
  ax = fig.add_subplot(gs[0,1])
  imshow(ax, grism); ax.set_anchor('W')
  fig.tight_layout()
  return fig
def show_flt_model(flt, figsize=(10,10)):
  z = ZScaleInterval()
  filter = flt.grism.filter
  f = flt.grism_file
  PA = flt.dispersion_PA
  def imshow(ax, data):
    vmin,vmax = z.get_limits(data)
    extent = [0, data.shape[0], 0, data.shape[1]]
    kwargs = dict(
        origin='lower',
        interpolation='Nearest',
        cmap=plt.cm.viridis,
        aspect=data.shape[1]/data.shape[0],
    )
    ax.imshow(data,  vmin=vmin, vmax=vmax, extent=extent, **kwargs)
    ax.set(xticklabels=[], yticklabels=[])
    ax.xaxis.set_tick_params(length=0); ax.yaxis.set_tick_params(length=0)
  model    = flt.model  
  fig = plt.figure(constrained_layout=True, figsize=figsize)
  gax = fig.add_subplot(111)
  gax.set(xticks=[], yticks=[])
  gax.set_title('{f} filter:{filter} PA:{PA}'.format(**locals()))
  gax.set_aspect(model.shape[1]/model.shape[0])
  
  gs = plt.GridSpec(2, 2, wspace=0, hspace=0)
  grism    = flt.direct.data['SCI']
  direct   = flt.direct.data[flt.direct.thumb_extension]
  residual = flt.direct.data['SCI'] - flt.model
  ax = fig.add_subplot(gs[0,0])
  imshow(ax, model); ax.set_anchor('E')
  ax = fig.add_subplot(gs[0,1])
  imshow(ax, grism); ax.set_anchor('W')
  ax = fig.add_subplot(gs[1,0])
  imshow(ax, direct); ax.set_anchor('E')
  ax = fig.add_subplot(gs[1,1])
  imshow(ax, residual); ax.set_anchor('W')
  fig.tight_layout()
  return fig
def show_id_beam(id, fltid, grp, ds9=False, todos=['image', 'grism', 'beams', '1d', 'decontam', 'gbeam'], show_beams=True, show_grism=True, allbeams=False):
  from matplotlib.patches import Rectangle
  z = ZScaleInterval()
  flt = grp.FLTs[fltid]
  title = "filter:{}, PA:{}, flt:{}".format(flt.grism.filter, flt.dispersion_PA, flt.grism_file.split('.fits')[0] )
  def imshow(data, figsize=(5,5), vmin=None, vmax=None):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    ax = fig.add_subplot(111)
    if vmin is None:
      vmin,vmax = z.get_limits(data)
    extent = [0, data.shape[1], 0, data.shape[0]]
    kwargs = dict(
        origin='lower',
        interpolation='Nearest',
        cmap=plt.cm.viridis,
        extent=extent,
        aspect='equal',
    )
    ax.imshow(data,  vmin=vmin, vmax=vmax, **kwargs)
    ax.set(xticklabels=[], yticklabels=[])
    ax.xaxis.set_tick_params(length=0); ax.yaxis.set_tick_params(length=0)
    return ax
  def imshow2(datas, figsize=(5,5), vmin=None, vmax=None):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(1, 2, wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    for ax,data in zip([ax0, ax1], datas):
      vmin,vmax = z.get_limits(data)
      extent = [0, data.shape[1], 0, data.shape[0]]
      kwargs = dict(
          origin='lower',
          interpolation='Nearest',
          cmap=plt.cm.viridis,
          extent=extent,
          aspect='equal',
      )
      ax.imshow(data,  vmin=vmin, vmax=vmax, **kwargs)
      ax.set(xticklabels=[], yticklabels=[])
      ax.xaxis.set_tick_params(length=0); ax.yaxis.set_tick_params(length=0)
    ax0.set_anchor('E')
    ax1.set_anchor('W')
    return [ax0, ax1]
  def imshowb(data, figsize=(5,5), vmin=None, vmax=None, bname=''):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(2, 1, wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0])
    extent = [0, data.shape[1], 0, data.shape[0]]
    kwargs = dict(
        origin='lower',
        interpolation='Nearest',
        cmap=plt.cm.viridis,
        aspect='equal',
        extent=extent,
    )
    _vmin,_vmax = z.get_limits(data)

    ax0.imshow(data,  vmin=vmin,  vmax=vmax, **kwargs)
    ax1.imshow(data,  vmin=_vmin, vmax=_vmax, **kwargs)
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.xaxis.set_tick_params(length=0);
    ax0.yaxis.set_tick_params(length=0)
    ax1.set(xticklabels=[], yticklabels=[])
    ax1.xaxis.set_tick_params(length=0);
    ax1.yaxis.set_tick_params(length=0)
    ax0.set_anchor('S')
    ax1.set_anchor('N')
    ax0.set_ylabel(bname)
    ax1.set_ylabel(bname)
    fig.tight_layout()
  def imshowb1(data, figsize=(5,5), vmin=None, vmax=None, bname='', *, wave, flux):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[:,1])
    extent = [0, data.shape[1], 0, data.shape[0]]
    kwargs = dict(
        origin='lower',
        interpolation='Nearest',
        cmap=plt.cm.viridis,
        aspect='equal',
        extent=extent,
    )
    _vmin,_vmax = z.get_limits(data)

    ax0.imshow(data,  vmin=vmin,  vmax=vmax, **kwargs)
    ax1.imshow(data,  vmin=_vmin, vmax=_vmax, **kwargs)
    ax0.set(xticklabels=[], yticklabels=[])
    ax0.xaxis.set_tick_params(length=0);
    ax0.yaxis.set_tick_params(length=0)
    ax1.set(xticklabels=[], yticklabels=[])
    ax1.xaxis.set_tick_params(length=0);
    ax1.yaxis.set_tick_params(length=0)
    ax0.set_anchor('S')
    ax1.set_anchor('N')
    ax0.set_ylabel(bname)
    ax1.set_ylabel(bname)
    ax2.plot(wave, flux)
    fig.tight_layout()
  def imshowc(sci, res, res_beam, contam, figsize=(15,8), *, y0, y1):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(2, 4, wspace=0, hspace=0)
    ax00 = fig.add_subplot(gs[0,0])
    ax01 = fig.add_subplot(gs[0,1])
    ax02 = fig.add_subplot(gs[0,2])
    ax03 = fig.add_subplot(gs[0,3])
    ax10 = fig.add_subplot(gs[1,0])
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])
    ax13 = fig.add_subplot(gs[1,3])
    extent = [0, sci.shape[1], 0, sci.shape[0]]
    kwargs = dict(
        origin='lower',
        interpolation='Nearest',
        cmap=plt.cm.viridis,
        aspect='equal',
        extent=extent,
    )
    vmin,vmax   = z.get_limits(sci)
    _vmin,_vmax = z.get_limits(contam)


    ax00.imshow(sci,    vmin=vmin, vmax=vmax, **kwargs)
    ax01.imshow(res_beam,    vmin=vmin, vmax=vmax, **kwargs)
    ax02.imshow(contam, vmin=vmin, vmax=vmax, **kwargs)
    ax03.imshow(res, vmin=vmin, vmax=vmax, **kwargs)
    ax10.imshow(sci,    vmin=_vmin, vmax=_vmax, **kwargs)
    ax11.imshow(res_beam,    vmin=_vmin, vmax=_vmax, **kwargs)
    ax12.imshow(contam, vmin=_vmin, vmax=_vmax, **kwargs)
    ax13.imshow(res,    vmin=_vmin, vmax=_vmax, **kwargs)
    ax00.set_ylabel('use zscale(SCI)')
    ax10.set_ylabel('use zscale(residual)')
    ax00.set_title('SCI')
    ax01.set_title('residual + beam model')
    ax02.set_title('contam model')
    ax03.set_title('residual(SCI-beam model-contam model)')
    for i, ax in enumerate([ax00,ax01,ax02,ax03,ax10,ax11,ax12,ax13]):
      ax.axhline(y0, 0, sci.shape[1], color='red')
      ax.axhline(y1, 0, sci.shape[1], color='red')
      ax.set(xticklabels=[], yticklabels=[])
      ax.xaxis.set_tick_params(length=0);
      ax.yaxis.set_tick_params(length=0)
      if i>3:
        ax.set_anchor('N')
      else:
        ax.set_anchor('S')
    fig.tight_layout()
  def imshowgb(gbeam, figsize=(15,15)):
    if gbeam is None:
      print('cat not get beam in this FLT!')
      return
    sci = gbeam.grism['SCI']
    res = sci - gbeam.model - gbeam.contam
    res_beam = sci - gbeam.contam
    contam = gbeam.contam

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(4, 1, wspace=0, hspace=0)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[2,0])
    ax3 = fig.add_subplot(gs[3,0])
    ax0.set_title("use zscale(SCI)")
    ax0.set_ylabel("SCI")
    ax1.set_ylabel("residual + beam model")
    ax2.set_ylabel("contam model")
    ax3.set_ylabel("residual (SCI-beam model-contam model)")
    extent = [0, sci.shape[1], 0, sci.shape[0]]
    kwargs = dict(
        origin='lower',
        interpolation='Nearest',
        cmap=plt.cm.viridis,
        aspect='equal',
        extent=extent,
    )
    vmin,vmax   = z.get_limits(sci)
    _vmin,_vmax = z.get_limits(contam)

    ax0.imshow(sci,      vmin=vmin, vmax=vmax, **kwargs)
    ax1.imshow(res_beam, vmin=vmin, vmax=vmax, **kwargs)
    ax2.imshow(contam,   vmin=vmin, vmax=vmax, **kwargs)
    ax3.imshow(res,      vmin=vmin, vmax=vmax, **kwargs)
    for ax in [ax0,ax1,ax2,ax3]:
      ax.set(xticklabels=[], yticklabels=[])
      ax.xaxis.set_tick_params(length=0);
      ax.yaxis.set_tick_params(length=0)
      #ax.set_anchor('E')
    fig.tight_layout()
  def plot_all_beams(gbeams, PAs, filters):
    z = ZScaleInterval()
    count = len(gbeams)

    NX = 1
    NY = count
    widths = []
    for i in range(NX):
        widths.extend([0.2, 0.2, 0.2, 1, 1])
    gs = plt.GridSpec(NY, NX*5, height_ratios=[1]*NY, width_ratios=widths)
    groups = {}
    fig = plt.figure(figsize=(13*NX, 1*NY))

    for i in range(count):
        gbeam = gbeams[i]
        if gbeam is None: continue
        rootname = gbeam.grism.header['ROOTNAME']
        grism_name = filters[rootname]
        pa = PAs[rootname]
        if grism_name not in groups:
            groups[grism_name] = []
        fname = "{}\n{}\n{:04d}\n{}".format(grism_name, pa, i, rootname)
        i_ref = gbeam.beam.direct
        i_seg = gbeam.beam.seg == gbeam.id
        s_sci = gbeam.grism.data["SCI"] - gbeam.contam
        s_con = gbeam.contam

        groups[grism_name].append([i_ref, i_seg, s_sci, s_con, fname, pa])
    for grism in groups:
        groups[grism].sort(key=lambda _:_[-1])

    c = -1
    for grism in groups:
        for each in groups[grism]:
            c += 1
            i_ref, i_seg, s_sci, s_con, fname, pa = each

            ax = fig.add_subplot(gs[c, 0])
            ax.text(-0.8, -0.8, fname)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.xaxis.set_tick_params(length=0)
            ax.yaxis.set_tick_params(length=0)
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            for j, todo in enumerate([i_ref, i_seg, s_sci, s_con]):
                ax = fig.add_subplot(gs[c, j+1])
                if c==0:
                  if j==0:
                    ax.set_title('REF')
                  elif j==1:
                    ax.set_title('seg')
                  elif j==2:
                    ax.set_title('SCI-contam model')
                  elif j==3:
                    ax.set_title("residual(SCI-contam model-beam model)")
                sh = todo.shape
                extent = [0, sh[1], 0, sh[0]]
                if j==1:
                    vmin = 0
                    vmax = 1
                elif j==3: # s_con use s_sci's limits
                    pass
                else:
                    vmin,vmax = z.get_limits(todo)
                ax.imshow(todo, origin='lower', interpolation='Nearest',
                        vmin=vmin, vmax=vmax, cmap=plt.cm.viridis,
                        extent=extent, aspect='auto')
                ax.set_xticklabels([]); ax.set_yticklabels([])
                ax.xaxis.set_tick_params(length=0)
                ax.yaxis.set_tick_params(length=0)
    gs.tight_layout(fig, pad=0)

  PAs = {}
  filters = {}
  for _filter in grp.PA:
    pas = grp.PA[_filter]
    for PA in pas:
      indexs = pas[PA]
      for index in indexs:
        _flt = grp.FLTs[index]
        PAs[_flt.grism.header['ROOTNAME']] = PA
        filters[_flt.grism.header['ROOTNAME']] = _filter
  wave_min = np.inf
  wave_max = -np.inf
  residual_beam = flt.grism['SCI'] - flt.model
  residual = flt.grism['SCI'] - flt.model
  contam = flt.model.copy()
  if ds9:
    from pyds9 import DS9
    ds9 = DS9()
    ds9.set('lock frame image')
    ds9.set('frame 1')
    ds9.set_np2arr(flt.seg)
    ds9.set('frame 2')
    ds9.set_fits(fits.HDUList([fits.PrimaryHDU(flt.direct.data[flt.direct.thumb_extension], header=flt.direct.header)]))
    ds9.set('frame 3')
    ds9.set_np2arr(flt.model)
    ds9.set('frame 1')
  if id is None:
    return
  size = 50
  beams = flt.compute_model_orders(id=id, get_beams=['A','B','C','D','E'], in_place=False, compute_size=size)
  gbeams = grp.get_all_beams(id, size=size, beam_id='A')
  gbeam = gbeams[fltid]
  if beams:
    for i, bname in enumerate(beams):
      beam = beams[bname]
      if i==0:
        dx,dy = beam.direct.shape[0]/2, beam.direct.shape[1]/2
        x,y = beam.xc, beam.yc
        if 'image' in todos and 'grism' in todos:
          zscale_imshow([[beam.direct, beam.seg==id]], title=title)
          ax_image, ax_grism = imshow2(
            [
              flt.direct.data[flt.direct.thumb_extension],
              flt.model,
            ],
            figsize=(40,20)
          )
          ax_grism.add_patch(Rectangle((x-dx, y-dy), dx*2, dy*2, color='red', fc='none', lw=2))
        elif 'image' in todos:
          zscale_imshow([[beam.direct, beam.seg==id]], title=title)
          ax_image = imshow(flt.direct.data[flt.direct.thumb_extension], figsize=(20,20))
        elif 'grism' in todos and show_grism:
          ax_grism = imshow(flt.model, figsize=(20,20))
          ax_grism.add_patch(Rectangle((x-dx, y-dy), dx*2, dy*2, color='red', fc='none', lw=2))
        vmin,vmax = z.get_limits(flt.model)
        if 'image' in todos:
          ax_image.add_patch(Rectangle((x-dx, y-dy), dx*2, dy*2, color='red', fc='none', lw=2))
        if ds9:
          ds9.set('frame 1')
          ds9.set('pan to {} {}'.format(x, y))
          rcommand = '''regions command 'box {x} {y} {dxx} {dyy} 0 # color=red' '''.format(
            dxx=dx*2,dyy=dy*2,**locals()
          )
          ds9.set(rcommand)
          ds9.set('frame 2')
          ds9.set(rcommand)
          ds9.set('frame 3')
          ds9.set(rcommand)
          ds9.set('scale limits {} {}'.format(vmin, vmax))
        wave_min = beam.lam_beam.min()
        wave_max = beam.lam_beam.max()
      
      sy = beam.sly_parent.start; ey = beam.sly_parent.stop
      sx = beam.slx_parent.start; ex = beam.slx_parent.stop
      tx, ty = (sx+ex)/2, (sy+ey)/2
      text = "{}({:.1f})".format(bname, (sx+ex)/2-x)
      if 'decontam' in todos:
        if i==0:
          residual_beam[beam.sly_parent, beam.slx_parent] = residual[beam.sly_parent, beam.slx_parent] + beam.model
          contam[beam.sly_parent, beam.slx_parent] = contam[beam.sly_parent, beam.slx_parent] - beam.model
          pad = 100
          xstart = tx - pad
          xend   = tx + pad
          ystart = ty - pad
          yend   = ty + pad
          if xstart<0: xstart=0
          if xend>flt.model.shape[1]-1: xend=flt.model.shape[1]-1
          if ystart<0: ystart=0
          if yend>flt.model.shape[0]-1: yend=flt.model.shape[0]-1
          xstart = int(xstart)
          ystart = int(ystart)
          xend = int(xend)
          yend = int(yend)
          y0 = beam.sly_parent.start; y0 += pad-ty
          y1 = beam.sly_parent.stop;  y1 += pad-ty
          if show_grism:
            ax_grism.add_patch(Rectangle((xstart, ystart), pad*2+1, pad*2+1, color='red', fc='none', lw=2))

        if i==0:
          beam0  = beam
          bname0 = bname
      if 'grism' in todos:
        if (show_beams or i==0) and not '1d' in todos:
          imshowb(beam.model, vmin=vmin, vmax=vmax, figsize=(15,5), bname=bname)
        if show_grism:
          ax_grism.add_patch(Rectangle((sx, sy), ex-sx, ey-sy, color='red', fc='none', lw=2))
          if 0<tx<flt.model.shape[0]:
            ax_grism.text(tx, ty, text, color='red')
        if ds9:
          rcommand = '''regions command 'box {x} {y} {dxx} {dyy} 0 # color=red' '''.format(
            dxx=ex-sx,dyy=ey-sy, x=(sx+ex)/2, y=(sy+ey)/2,
          )
          ds9.set(rcommand)
          rcommand = '''regions command "text {x} {y} '{text}' # color=red" '''.format(
            x=tx, y=ty, text=text
          )
          ds9.set(rcommand)
      #ax = imshow(a.modelf.reshape(a.sh_beam), kwargs_fig=dict(figsize=(15,20)))
      #ax = imshow(flt.model[a.sly_parent, a.slx_parent], kwargs_fig=dict(figsize=(15,20)))
      #ax = imshow(flt.grism['SCI'][a.sly_parent, a.slx_parent], kwargs_fig=dict(figsize=(15,20)))
      #ax = imshow(flt.model) 
  if '1d' in todos:
    _,(wave, flux),_ = flt.object_dispersers[id] 
    mask = (wave_min < wave) & (wave < wave_max)
    #mask = np.ones(wave.shape).astype(np.bool)
    if show_beams:
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.plot(wave[mask], flux[mask])
    else:
      imshowb1(beam0.model, vmin=vmin, vmax=vmax, figsize=(15,5), bname=bname0, wave=wave[mask], flux=flux[mask])
  if 'decontam' in todos:
    imshowc(
      flt.grism['SCI'][ystart:yend, xstart:xend],
      residual[ystart:yend, xstart:xend],
      residual_beam[ystart:yend, xstart:xend],
      contam[ystart:yend, xstart:xend],
      y0=y0, y1=y1,
      )
  if 'gbeam' in todos:
    imshowgb(gbeam)
  if allbeams:
    plot_all_beams(gbeams, PAs, filters)
  result = {'beams': beams, 'gbeams': gbeams}
  if ds9:
    result['ds9'] = ds9
  return result

def plot_all_beams(filename, save=True, show=False, type="res", figsize=13):
    from matplotlib.gridspec import GridSpec
    # filename = 'j071732p3745_00361.beams.fits'
    z = ZScaleInterval()
    outfile = filename[:-5]+'.png'
    obj = fits.open(filename)
    count = obj[0].header['COUNT']
    #beams_dir = filename[:-5]
    #os.makedirs(beams_dir, exist_ok=True)

    NX = 1
    NY = count
    widths = []
    for i in range(NX):
        widths.extend([0.2, 0.2, 0.2, 1, 1])
    gs = GridSpec(
      NY, NX*5, height_ratios=[1]*NY, width_ratios=widths,
      wspace=0.02, hspace=0.02,
      )
    groups = {}
    ratio = NY/(13*NX)
    fig = plt.figure(figsize=(figsize, figsize*ratio))

    for i in range(count):
        flt_file_name = obj[0].header['FILE{:04d}'.format(i)]
        grism_name = obj[0].header['GRIS{:04d}'.format(i)]
        pa = obj[0].header['PA{:04d}'.format(i)]
        if grism_name not in groups:
            groups[grism_name] = []
        fname = "{}\n{}\n{:04d}\n{}".format(grism_name, pa, i, flt_file_name.split('_')[0])
        i_ref = obj[i*7+1].data
        i_seg = obj[i*7+2].data
        s_sci = obj[i*7+3].data
        s_con = obj[i*7+6].data
        if type=='res':
          s_sci = s_sci - s_con
        elif type=='sci':
          pass
        else:
          raise Exception('unknown type')

        groups[grism_name].append([i_ref, i_seg, s_sci, s_con, fname, pa])
    for grism in groups:
        groups[grism].sort(key=lambda _:_[-1])

    c = -1
    for grism in groups:
        for each in groups[grism]:
            c += 1
            i_ref, i_seg, s_sci, s_con, fname, pa = each

            ax = fig.add_subplot(gs[c, 0])
            ax.text(-0.8, -0.8, fname)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.xaxis.set_tick_params(length=0)
            ax.yaxis.set_tick_params(length=0)
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])

            for j, data in enumerate([i_ref, i_seg, s_sci, s_con]):
                ax = fig.add_subplot(gs[c, j+1])
                sh = data.shape
                extent = [0, sh[1], 0, sh[0]]
                if j==1:
                    vmin = data.min()
                    vmax = data.max()
                elif j==3: # s_con use s_sci's limits
                    pass
                else:
                    vmin,vmax = z.get_limits(data)
                ax.imshow(data, origin='lower', interpolation='Nearest',
                        vmin=vmin, vmax=vmax, cmap=plt.cm.viridis,
                        extent=extent, aspect='auto')
                ax.set_xticklabels([]); ax.set_yticklabels([])
                ax.xaxis.set_tick_params(length=0)
                ax.yaxis.set_tick_params(length=0)
    gs.tight_layout(fig, pad=0)
    if save:
      fig.savefig(outfile)
    if not show:
      plt.close(fig)
def plot_id_output(id, *, root_name, limits=None, models=['R30', '1D', 'TEMP', '2D', 'redshift'], only_flux=False, figsize=None):
  result = {}
  name = '{0}_{1:05d}.full.fits'.format(root_name, id)
  if os.path.exists(name) and 'redshift' in models:
    from astropy.table import Table
    tfit_fits = ff  = fits.open(name)
    redshift =  ff['ZFIT_STACK'].header['Z_MAP']
  else:
    tfit_fits=None
    redshift=0

  if '2D' in models:
    name = '{0}_{1:05d}.stack.fits'.format(root_name, id)
    fstack = fits.open(name)
    keys = filter(lambda _:_.startswith('GRISM'), fstack[0].header.keys())
    grisms = list(map(lambda _:fstack[0].header[_], keys))
    names = []
    PAs = {}
    nG = len(grisms)
    nP = 0
    height_ratios=[]
    width_ratios=[]
    for grism in grisms:
      keys = filter(lambda _:_.startswith(grism), fstack[0].header.keys())
      pas = list(map(lambda _:fstack[0].header[_], keys))
      PAs[grism] = pas
      if len(pas)>nP: nP = len(pas)
    for i, grism in enumerate(PAs):
      gshape = fstack['SCI', grism].data.shape
      if i==0:
        height_ratios = [gshape[0]]*(nP+2) + [3*gshape[0]]
      width_ratios.extend([gshape[0], gshape[1]])
      for pa in PAs[grism]:
        pass
    ratio = np.sum(height_ratios)/np.sum(width_ratios)
    fig = plt.figure(figsize=(figsize, figsize*ratio))
    gs = plt.GridSpec(len(height_ratios), len(width_ratios), width_ratios=width_ratios, height_ratios=height_ratios, hspace=0.01, wspace=0.01)
    z = ZScaleInterval()
    todo_axis = []
    axis_x_limis = []
    wave_ranges = {}
    bprops = dict(boxstyle='round', facecolor='white', alpha=1)

    for i, grism in enumerate(PAs):
      npas = len(PAs[grism])
      nrow = nP - npas - 1
      for j, pa in enumerate(PAs[grism]):
        nrow += 1
        ax0=ax_kernel   = fig.add_subplot(gs[nrow, i*2+0])
        ax1=ax_spectrum = fig.add_subplot(gs[nrow, i*2+1])
        data_kernel   = fstack['KERNEL', "{},{}".format(grism, pa)].data
        data_spectrum = fstack['SCI', "{},{}".format(grism, pa)].data
        ninput = fstack['SCI', "{},{}".format(grism, pa)].header['NINPUT']
        vmin,vmax = z.get_limits(data_spectrum)
        ax_kernel.imshow(
          data_kernel,
          cmap=plt.cm.viridis, aspect='auto', origin='lower', interpolation='Nearest',
          extent=[0,data_kernel.shape[1], 0, data_kernel.shape[0]],
         )
        ax_spectrum.imshow(
          data_spectrum,
          cmap=plt.cm.viridis, aspect='auto', origin='lower', interpolation='Nearest',
          extent=[0,data_spectrum.shape[1], 0, data_spectrum.shape[0]],
          vmin=vmin, vmax=vmax,
         )
        ax_spectrum.text(0.02, 0.95, "{}:{}({})".format(grism, pa, ninput), transform=ax_spectrum.transAxes, verticalalignment='top', bbox=bprops)
        if i==0 and j==0:
          ax_kernel.text(0.02, 0.95, "id={}".format(id), transform=ax_kernel.transAxes, verticalalignment='top', bbox=bprops)

        ax0.set_xticklabels([]); ax0.set_yticklabels([]); ax0.xaxis.set_tick_params(length=0); ax0.yaxis.set_tick_params(length=0)
        ax1.set_xticklabels([]); ax1.set_yticklabels([]); ax1.xaxis.set_tick_params(length=0); ax1.yaxis.set_tick_params(length=0)
      nrow += 1
      ax0=ax_kernel   = fig.add_subplot(gs[nrow, i*2+0])
      ax1=ax_spectrum = fig.add_subplot(gs[nrow, i*2+1])

      data_kernel   = fstack['KERNEL', "{}".format(grism)].data
      data_spectrum = fstack['SCI', "{}".format(grism)].data
      ninput = fstack['SCI', "{}".format(grism)].header['NINPUT']
      vmin,vmax = z.get_limits(data_spectrum)
      ax_kernel.imshow(
        data_kernel,
        cmap=plt.cm.viridis, aspect='auto', origin='lower', interpolation='Nearest',
        extent=[0,data_kernel.shape[1], 0, data_kernel.shape[0]],
        )
      ax_spectrum.imshow(
        data_spectrum,
        cmap=plt.cm.viridis, aspect='auto', origin='lower', interpolation='Nearest',
        extent=[0,data_spectrum.shape[1], 0, data_spectrum.shape[0]],
        vmin=vmin, vmax=vmax,
        )
      ax_spectrum.text(0.02, 0.95, '{}({})'.format(grism, ninput), transform=ax_spectrum.transAxes, verticalalignment='top', bbox=bprops)
      ax0.set_xticklabels([]); ax0.set_yticklabels([]); ax0.xaxis.set_tick_params(length=0); ax0.yaxis.set_tick_params(length=0)
      ax1.set_xticklabels([]); ax1.set_yticklabels([]); ax1.xaxis.set_tick_params(length=0); ax1.yaxis.set_tick_params(length=0)

      nrow += 1
      ax0=ax_kernel   = fig.add_subplot(gs[nrow, i*2+0])
      ax1=ax_spectrum = fig.add_subplot(gs[nrow, i*2+1])

      data_kernel   = fstack['KERNEL', "{}".format(grism)].data
      data_spectrum = fstack['SCI', "{}".format(grism)].data - fstack['MODEL', "{}".format(grism)].data
      vmin,vmax = z.get_limits(data_spectrum)
      ax_kernel.imshow(
        data_kernel,
        cmap=plt.cm.viridis, aspect='auto', origin='lower', interpolation='Nearest',
        extent=[0,data_kernel.shape[1], 0, data_kernel.shape[0]],
        )
      ax_spectrum.imshow(
        data_spectrum,
        cmap=plt.cm.viridis, aspect='auto', origin='lower', interpolation='Nearest',
        extent=[0,data_spectrum.shape[1], 0, data_spectrum.shape[0]],
        vmin=vmin, vmax=vmax,
        )
      ax_spectrum.text(0.02, 0.95, '{}({}): sci - model'.format(grism,ninput), transform=ax_spectrum.transAxes, verticalalignment='top', bbox=bprops)
      ax0.set_xticklabels([]); ax0.set_yticklabels([]); ax0.xaxis.set_tick_params(length=0); ax0.yaxis.set_tick_params(length=0)
      ax1.set_xticklabels([]); ax1.set_yticklabels([]); ax1.xaxis.set_tick_params(length=0); ax1.yaxis.set_tick_params(length=0)

      nrow += 1
      ax = fig.add_subplot(gs[nrow, i*2+1])
      axis_x_limis.append((fstack['SCI', grism].header['WMIN']*10000, fstack['SCI', grism].header['WMAX']*10000,))
      ax.set_xlabel("$\lambda(\AA)$-{}\nz={:.2f}".format(grism, redshift))
      todo_axis.append(ax)
  else:
    ratio = 3/5
    fig, ax = plt.subplots(1, figsize=(figsize, figsize*ratio))
    todo_axis = [ax]

  name = '{0}_{1:05d}.beams-fit.fits'.format(root_name, id)
  if os.path.exists(name):
    mb = multifit.MultiBeam(name, fcontam=0.2, group_name=root_name, psf=False, min_sens=0.05, verbose=False)
    mbplots_sci  = extract_single_beams(mb, tfit_fits=tfit_fits)
  else:
    mb = None
    mbplots_sci = None

  for ax in todo_axis:
    colors = { "G102": 'green', "G141": 'orange', }
    ls = { "G102": '--', "G141": '-.', }
    name = '{0}_{1:05d}.R30.fits'.format(root_name, id)
    if os.path.exists(name) and 'R30' in models:
      f30 = fits.open(name)
      for ext in f30[1:]:
        fname = ext.header['EXTNAME']
        d = f30[fname].data
        wave,flux,cont,line,flat,err = d['wave'], d['flux'], d['cont'], d['line'], d['flat'], d['err']
        ax.scatter(wave, flux/flat, color=colors[fname], label='{} R30'.format(fname), marker="^")
        ax.errorbar(wave, flux/flat, err/flat, ls='none', color=colors[fname],)
    name = '{0}_{1:05d}.full.fits'.format(root_name, id)
    if os.path.exists(name) and 'TEMP' in models:
      ff  = fits.open(name)
      temp = ff['TEMPL'].data
      wave, cont, full = temp['wave'], temp['continuum'], temp['full']
      mask = (wave>8000)&(wave<17000)
      ax.plot(wave[mask], full[mask], alpha=0.4, color='cyan', label='temp cont+line (1D model)', ls=':', lw=4)
      ax.plot(wave[mask], cont[mask], alpha=0.4, color='brown',  label='temp cont (1D model)', ls='-', lw=4)
    name = '{0}_{1:05d}.1D.fits'.format(root_name, id)
    if os.path.exists(name) and '1D' in models:
      f1d = fits.open(name)
      for ext in f1d[1:]:
        fname = ext.header['EXTNAME']
        d = f1d[fname].data
        wave,flux,cont,line,flat,err = d['wave'], d['flux'], d['cont'], d['line'], d['flat'], d['err']
        #ax.plot(wave, flux/flat, color=colors[fname], label='{} flux'.format(fname), ls='-', lw=2, alpha=0.5)
        ax.scatter(wave, flux/flat, color=colors[fname], label='{} flux (data)'.format(fname), marker='.', alpha=0.3, s=50)
        ax.errorbar(wave, flux/flat, err/flat, color=colors[fname], alpha=0.3, ls='none')
        if not only_flux:
          ax.plot(wave, cont/flat, color=colors[fname], label='{} cont (2D model)'.format(fname), ls='--', lw=3)
          ax.plot(wave, (line)/flat, color=colors[fname], label='{} cont+line (2D model)'.format(fname), ls=':', lw=3)
    if mbplots_sci is not None:
      for i, (wave, flux, err) in enumerate(mbplots_sci):
        if i==0:
          ax.errorbar(wave, flux, err, alpha=0.5/mb.N, color='k', label='single sci')
        else:
          ax.errorbar(wave, flux, err, alpha=0.5/mb.N, color='k')
      #for i, (wave, flux, err) in enumerate(mbplots_model):
      #  ax.scatter(wave, flux, err, alpha=1/mb.N, color='red')
      #  if i==0:
      #    ax.errorbar(wave, flux, err, alpha=1/mb.N, color='red', label='single model')
      #  else:
      #    ax.errorbar(wave, flux, err, alpha=1/mb.N, color='red')
    ax.legend()
    ax.set_ylabel(r'$f_\lambda$ erg s$^{-1}$ cm$^{-2}$ $\AA$')
    if limits:
      ax.set_ylim(limits)
  if '2D' in models:
    for lim,ax in zip(axis_x_limis, todo_axis):
      ax.set_xlim(*lim)
      axu = ax.twiny()
      axu.tick_params(axis="x",direction="in", pad=-15)
      xticks = copy.deepcopy(ax.get_xticks())
      xlim = ax.get_xlim()
      for i in range(len(xticks)):
        xticks[i] /= (1+redshift)
      axu.set_xticks(xticks)
      axu.set_xlim(xlim[0]/(1+redshift), xlim[1]/(1+redshift))

  if tfit_fits is not None:
    from astropy.table import Table
    rfig, rax = plt.subplots(1)
    # same as the fit table above, redshift fit to the stacked spectra
    fit_stack = Table(ff['ZFIT_STACK'].data) 
    rax.plot(fit_stack['zgrid'], fit_stack['pdf'], label='Stacked')
    # zoom in around the initial best-guess with the individual "beam" spectra
    if 'ZFIT_BEAM' in ff:
      fit_beam = Table(ff['ZFIT_BEAM'].data)   
      rax.plot(fit_beam['zgrid'], fit_beam['pdf'], label='Zoom, beams')
    tfit_fits = fits.open(name)
    line_keys = filter(lambda _:_.startswith('LINE'), ff[0].header.keys())
    lines = {}
    for key in line_keys:
      index = key[-3:]
      name = ff[0].header[key]
      flux = ff[0].header['FLUX'+index]
      err  = ff[0].header['ERR'+index]
      wave = ff["LINE", name].header['WAVELEN']
      restwave = ff["LINE", name].header['RESTWAVE']
      lines[name] = (wave, flux, err)
    for ax in todo_axis:
      ylim = ax.get_ylim()
      xlim = ax.get_xlim()
      deltay = ylim[1]-ylim[0]
      for line in lines:
        wave,flux,err = lines[line]
        if wave>xlim[1] or wave<xlim[0]: continue
        if flux>0 and err>0 and flux > 3*err:
          ax.vlines(wave, ylim[0], ylim[1], ls='--', color='red', lw=1.2, alpha=0.6)
          ax.text(wave, ylim[0]-deltay*0.2, line, rotation=90, verticalalignment='center', color='red')
        else:
          ax.vlines(wave, ylim[0], ylim[1], ls='--', color='black', lw=1, alpha=0.4)
          ax.text(wave, ylim[0]-deltay*0.2, line, rotation=90, verticalalignment='center')
  else:
    tfit_fits=None
    redshift=0

  fig.tight_layout()
  for each in [
      'redshift',
      'PAs',
      'fstack',
      'mb',
      'mbplots_sci',
      'f30',
      'f1d',
      'ff',
    ]:
    result[each] = locals().get(each)
  return result

def extract_single_beams(beams, tfit_fits=None):
  '''some of the code are from fitting.py/GroupFitter/oned_figure in the Grizli package'''
  sci   = []
  # model = []
  bin=1
  min_sens_show = 0.1
  mspl = None
  tfit = None
  if tfit_fits is not None:
    header = tfit_fits['TEMPL'].header
  else:
    header = None

  for i in range(beams.N):
    beam = beams.beams[i]
    b_mask = beam.fit_mask.reshape(beam.sh)
    
    #temp = tfit_fits['TEMPL'].data
    #wave, cont, full = temp['wave'], temp['continuum'], temp['full']
    #wave = wave.astype(np.float64)
    #full = full.astype(np.float64)
    #m_i = beam.compute_model(spectrum_1d=(wave, full), is_cgs=True, in_place=False).reshape(beam.sh)
    #w, flm, erm = beam.beam.optimal_extract(m_i, bin=bin, ivar=beam.ivar*b_mask)
          
    if mspl is not None:
        mspl_i = beam.compute_model(spectrum_1d=mspl, is_cgs=True, in_place=False).reshape(beam.sh)
        
    f_i = beam.flat_flam.reshape(beam.sh)*1
        
    if hasattr(beam, 'init_epsf'): # grizli.model.BeamCutout
        if beam.grism.instrument == 'NIRISS':
            grism = beam.grism.pupil
        else:
            grism = beam.grism.filter
        
        clean = beam.grism['SCI'] - beam.contam 
        if header is not None:
          bgname = 'bg {0:03d}'.format(i)
          cname = list(filter(lambda _:header[_] == bgname, header.keys()))[0]
          cindex = cname[-3:]
          ### print("{}:{:.5e} clean:{:.5e}".format(bgname, header['CVAL'+cindex], np.median(clean)), end='')
          clean -= header['CVAL'+cindex]

        if mspl is not None:
            w, flspl, erm = beam.beam.optimal_extract(mspl_i, bin=bin, ivar=beam.ivar*b_mask)
                
        w, fl, er = beam.beam.optimal_extract(clean, bin=bin, ivar=beam.ivar*b_mask)            
        w, sens, ers = beam.beam.optimal_extract(f_i, bin=bin, ivar=beam.ivar*b_mask)
        #sens = beam.beam.sensitivity                
    else:
        grism = beam.grism
        clean = beam.sci - beam.contam
        if header is not None:
            bgname = 'bg {0:03d}'.format(i)
            cname = list(filter(lambda _:header[_].startswith(bgname), header.keys()))[0]
            cindex = cname[-3:]
            clean -= header['CVAL'+cindex]
        
        if mspl is not None:
            w, flspl, erm = beam.beam.optimal_extract(mspl_i, bin=bin, ivar=beam.ivar*b_mask)
            
        w, fl, er = beam.optimal_extract(clean, bin=bin, ivar=beam.ivar*b_mask)            
        #w, flc, erc = beam.optimal_extract(beam.contam, bin=bin, ivar=beam.ivar*b_mask)            
        w, sens, ers = beam.optimal_extract(f_i, bin=bin, ivar=beam.ivar*b_mask)
        
        #sens = beam.sens
    
    sens[~np.isfinite(sens)] = 1
    pscale = 1.
                                          
    unit_corr = 1./sens # /1.e-19#/pscale
    clip = (sens > min_sens_show*sens.max()) 
    clip &= (er > 0)
    if clip.sum() == 0:
      continue
    ###print('fluxm:{:.5e}ivar:{:.5e}mask:{:.5e}sens:{:.5e}flat:{:.5e}'.format(
    ###    np.median(fl),
    ###    np.median(beam.ivar),
    ###    np.median(b_mask),
    ###    np.nanmedian(fl),
    ###    np.median(sens),
    ###    np.median(beam.flat_flam),
    ###))
    
    fl *= unit_corr/pscale#/1.e-19
    er *= unit_corr/pscale#/1.e-19

    #flm *= unit_corr/pscale#/1.e-19
    #erm *= unit_corr/pscale#/1.e-19
        
    wave = w[clip].astype(np.float64)
    flux = fl[clip].astype(np.float64)

    err  = er[clip].astype(np.float64)
    sci.append((wave, flux, err))
    #wave = w[clip].astype(np.float64)
    #flux = flm[clip].astype(np.float64)
    #err  = erm[clip].astype(np.float64)
    #model.append((wave, flux, err))
  return sci
        
def objfilter(data):
  thisfilter = lambda row: \
    any(each in row['instrument_name'] for each in  CSST_INSTRUMENTS) and \
    any(each in row['filters'] for each in CSST_UV_FILTERS) and \
    row['dataRights'] == 'PUBLIC'
  mask = [thisfilter(_) for _ in data]
  return data[mask]

# https://github.com/gbrammer/mastquery/blob/a886d4c4f332a81718ac363f3765a4d346c72ec2/mastquery/query.py
def parse_polygons(polystr):
    if hasattr(polystr, 'decode'):
        decoded = polystr.decode('utf-8').strip().upper()
    else:
        decoded = polystr.strip().upper()
    
    polyspl = decoded.replace('POLYGON','xxx').replace('CIRCLE','xxx')
    polyspl = polyspl.split('xxx')
    
    poly = []
    for pp in polyspl:
        if not pp:
            continue
            
        spl = pp.strip().split()
        for ip, p in enumerate(spl):
            try:
                pf = float(p)
                break
            except:
                continue
        
        
        try:
            poly_i = np.cast[float](spl[ip:]).reshape((-1,2)) #[np.cast[float](p.split()).reshape((-1,2)) for p in spl]
        except:
            # Circle
            x0, y0, r0 = np.cast[float](spl[ip:])
            cosd = np.cos(y0/180*np.pi)
            poly_i = np.array([XCIRCLE*r0/cosd+x0, YCIRCLE*r0+y0]).T
            
        ra = Angle(poly_i[:,0]*u.deg).wrap_at(360*u.deg).value
        poly_i[:,0] = ra
        if len(poly_i) < 2:
            continue
        
        poly.append(poly_i)
        
    return poly
def region2moc(regions, plot=False):
  moc = None
  for each in regions:
    polygons = parse_polygons(each)
    for ep in polygons:
      skycoord = SkyCoord(ep, unit="deg", frame='icrs')
      _moc = MOC.from_polygon_skycoord(skycoord, max_depth=16)
      if moc is None:
        moc = _moc
      else:
        moc = moc.union(_moc)
  return moc
def moc_plot(moc,
  fov=360*u.deg,
  center=SkyCoord(0, 0, unit='deg', frame='icrs'),
  rotation=Angle(0, u.degree),
  projection="AIT"
  ):
  import matplotlib.pyplot as plt
  fig = plt.figure(111, figsize=(15, 15))
  # Define a WCS as a context
  with World2ScreenMPL(fig, 
          fov=fov,
          center=center,
          coordsys="icrs",
          rotation=rotation,
          projection=projection) as wcs:
      ax = fig.add_subplot(1, 1, 1, projection=wcs)
      # Call border giving the matplotlib axe and the `~astropy.wcs.WCS` object.
      moc.border(ax=ax, wcs=wcs, alpha=0.5, color="red")
  plt.xlabel('ra')
  plt.ylabel('dec')
  plt.grid(color="black", linestyle="dotted")
# drizzle_grisms_and_PAs
'''
id = source_ids[0]

# Pull out the 2D cutouts
beams = grp.get_beams(id, size=80)
mb = multifit.MultiBeam(beams, fcontam=0.2, group_name=root, psf=False)

# Save a FITS file with the 2D cutouts (beams) from the individual exposures
mb.write_master_fits()

# Fit polynomial model for initial continuum subtraction
wave = np.linspace(2000,2.5e4,100)
poly_templates = grizli.utils.polynomial_templates(wave, order=7)
pfit = mb.template_at_z(z=0, templates=poly_templates, fit_background=True, 
                        fitter='lstsq', get_uncertainties=2)

# Drizzle grisms / PAs and make a figure
hdu, fig = mb.drizzle_grisms_and_PAs(fcontam=0.2, flambda=False, kernel='point', 
                                     size=32, zfit=pfit, diff=True)

# Save drizzle figure FITS file
fig.savefig('{0}_{1:05d}.stack.png'.format(root, id))
hdu.writeto('{0}_{1:05d}.stack.fits'.format(root, id), clobber=True)
'''
