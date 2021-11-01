from .tools import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.visualization import ZScaleInterval
import copy

def test():
  print('kuku')

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
def show_id_beam(id, fltid, grp, ds9=False, todos=['image', 'grism']):
  from matplotlib.patches import Rectangle
  z = ZScaleInterval()
  # print(flt.grism.filter)
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
  def imshow2(data, figsize=(5,5), vmin=None, vmax=None, bname=''):
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
  if ds9:
    from pyds9 import DS9
    ds9 = DS9()
    ds9.set('lock frame image')
    ds9.set('frame 1')
    ds9.set_np2arr(flt.seg)
    ds9.set('frame 2')
    ds9.set_np2arr(flt.direct.data[flt.direct.thumb_extension])
    ds9.set('frame 3')
    ds9.set_np2arr(flt.model)
    ds9.set('frame 1')
  beams = flt.compute_model_orders(id=id, get_beams=['A','B','C','D','E'], in_place=False)
  gbeams = grp.get_beams(id, size=size, beam_id='A', min_sens=min_sens)
  for i, bname in enumerate(beams):
    beam = beams[bname]
    if i==0:
      if 'image' in todos:
        zscale_imshow([[beam.direct, beam.seg==id]])
        ax_image = imshow(flt.direct.data[flt.direct.thumb_extension], figsize=(20,20))
      dx,dy = beam.direct.shape[0]/2, beam.direct.shape[1]/2
      x,y = beam.xc, beam.yc
      if 'grism' in todos:
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

    if 'grism' in todos:
      imshow2(beam.model, vmin=vmin, vmax=vmax, figsize=(15,5), bname=bname)
      sy = beam.sly_parent.start; ey = beam.sly_parent.stop
      sx = beam.slx_parent.start; ex = beam.slx_parent.stop
      ax_grism.add_patch(Rectangle((sx, sy), ex-sx, ey-sy, color='red', fc='none', lw=2))
      tx, ty = (sx+ex)/2, (sy+ey)/2
      text = "{}({:.1f})".format(bname, (sx+ex)/2-x)
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
  if ds9:
    return ds9