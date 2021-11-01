import astropy
import astropy.units as u
import astropy.io.fits as fits
from astropy.table import Table, Column, join, vstack, hstack, setdiff, unique
from astropy.coordinates import SkyCoord
from astropy.samp import SAMPIntegratedClient
from pathlib import Path
from astroquery.xmatch import XMatch
import numpy as np
import yaml
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import numpy as np
from scipy.optimize import curve_fit

from astroquery.mast import Observations
from astroquery.simbad import Simbad

import sqlite3
import pickle
import sys
import os
from shutil import copyfile
from functools import reduce
from scipy.interpolate import interp1d
import astropy.units as u
import time
from astropy.nddata import StdDevUncertainty
from astropy.modeling import models, fitting
from datetime import datetime
import multiprocessing
import logging
import traceback
logger = logging.getLogger()
logger.disabled = True

# for SAMP to communicate with aladin and topcat
class Receiver(object):
  def __init__(self, client):
    self.client = client
    self.count              = 0
    self.count_call         = 0
    self.count_notification = 0
    self.call         = []
    self.notification = []
  def receive_call(self, private_key, sender_id, msg_id, mtype, params, extra):
    self.call.append({
      'sender_id': sender_id,
      'msg_id': msg_id,
      'mtype': mtype,
      'params': params,
      'extra': extra,
    })
    self.count_call += 1
    self.count += 1
    self.client.reply(msg_id, {"samp.status": "samp.ok", "samp.result": {}})
  def receive_notification(self, private_key, sender_id, mtype, params, extra):
    self.notification.append({
      'sender_id': sender_id,
      'mtype': mtype,
      'params': params,
      'extra': extra,
    })
    self.count += 1
    self.count_notification += 1
  def __repr__(self):
    return('Receiver({})'.format(self.count))
class SAMP(object):
  @staticmethod
  def write_to_temp(table):
    temproot = Path("/tmp/astropy")
    os.makedirs('/tmp/astropy', exist_ok=True)
    h = np.abs(hash(str(table)))
    tempfile = temproot.joinpath(str(h)+'.vot')
    if not tempfile.exists():
      table.write(str(tempfile.absolute()), format='votable')
    return tempfile
  def __init__(self, name='astropy'):
    self.c = SAMPIntegratedClient(name=name)
    self.c.connect()
    self.r = Receiver(self.c)
    self.c.bind_receive_call("table.load.votable", self.r.receive_call)
    self.c.bind_receive_notification("table.load.votable", self.r.receive_notification)
    self._get_clients()
  def _get_clients(self):
    _clients = self.c.get_registered_clients()
    clients = {}
    for each in _clients:
      meta = self.c.get_metadata(each)
      if 'samp.name' in meta:
        clients[meta['samp.name']] = each
    self.clients = clients
  def __del__(self):
    print('SAMP: disconnect and delete')
    self.c.disconnect()
  def send_table(self, receiver=None, *, table, name):
    message = {}
    params = {}
    params["name"] = name
    if isinstance(table, str): # table is path of vo table
      params["url"] = 'file://localhost' + os.path.abspath(table)
    elif isinstance(table, astropy.table.table.Table):
      path = self.write_to_temp(table)
      params["url"] = 'file://localhost' + str(path.absolute())
    message["samp.mtype"] = "table.load.votable"
    message["samp.params"] = params
    #print('sending to {}\n{}'.format(receiver, message))
    if receiver is None:
      self.c.notify_all(message)
    else:
      self.c.notify(self.clients[receiver], message)
  def aladin_point_to(self, ra=None, dec=None, l=None, b=None, wait=30, sleep=1):
    if ra is not None and dec is not None:
      params = {
        'ra':  str(ra),
        'dec': str(dec),
      }
    elif l is not None and b is not None:
      coord = SkyCoord(frame="galactic", l=l, b=b, unit=(u.deg, u.deg))
      params = {
        'ra':  str(coord.gcrs.ra.value),
        'dec': str(coord.gcrs.dec.value),
      }
    else:
      raise Exception('bad input')
    message = {
      'samp.mtype': 'coord.pointAt.sky',
      'samp.params': params,
    }
    if wait:
      r = self.c.call_and_wait(self.clients['Aladin'], message, str(wait))
      time.sleep(sleep)
      return r
    else:
      return self.c.notify(self.clients['Aladin'], message)
  def aladin_script(self, script, wait=30, sleep=1):
    message = {
      'samp.mtype': 'script.aladin.send',
      'samp.params': {
          'script': script
        },
    }
    if wait:
      r = self.c.call_and_wait(self.clients['Aladin'], message, str(wait))
      time.sleep(sleep)
      return r
    else:
      return self.c.notify(self.clients['Aladin'], message)

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

def raDec2str(*, ra,dec):
  coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
  return coord2str(coord)
def coord2str(coord):
  return 'J{0}{1}'.format(
    coord.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True),
    coord.dec.to_string(sep='', precision=2, alwayssign=True, pad=True),)

class MultiMatch(object):
  '''match more than two catalogues'''
  name_type = "U50"
  def __init__(self, data, samp=None):
    '''
      input data format: {
        <name>: {
          table: TABLE,
          id:   'id_key'
          ra:   'ra_key',
          dec:  'dec_key',
          # key_maps: {
            # key_name: key_origin_name/lambda
          # },
          ra_unit:  'deg',
          dec_unit: 'deg',
        },
        ...
      }
      output data format:
      Table with
      id, ra, dec, name1, name2, ....
    '''
    self.samp = samp
    self.raw_data = data
    self._init_table()
    self._do_crossmatch()
  def _init_table(self):
    self._number_sum = 0
    columns = [
      Column(name='id',  dtype=self.name_type),
      Column(name='count',  dtype=np.int),
      Column(name='ra',  dtype=np.float),
      Column(name='dec', dtype=np.float),
      ]
    # add ra,dec,coord and _id to raw_data
    for name in self.raw_data:
      __ = self.raw_data[name]
      table = __['table']
      self._number_sum += len(table)
      key_id  = __.get('id',   'id')
      key_ra  = __.get('ra',   'ra')
      key_dec = __.get('dec', 'dec')
      unit_ra  = __.get('unit_ra',  'deg')
      unit_dec = __.get('unit_dec', 'deg')
      print("{:5d} {}: {}({})=>ra {}({})=dec".format(len(table), name, key_ra, unit_ra, key_dec, unit_dec))
      coord = SkyCoord(ra=table[key_ra], dec=table[key_dec], unit=(unit_ra, unit_dec))
      table['ra']  = coord.ra.to(u.deg).value
      table['dec'] = coord.dec.to(u.deg).value
      table['coord'] = coord
      table['_id'] = table[key_id]
      # if self.samp is not None:
        # self.samp.send_table(name='M:'+name, table=table)
      table.sort(['dec', 'ra'])
      columns.append(Column(name='M:'+name, dtype=self.name_type))
    columns_with_name = {_.name:_ for _ in columns}
    #self.matched = Table(columns_with_name, masked=True)
    self.matched = Table(columns_with_name)
  def _do_crossmatch(self, radius=5.0/3600):
    tables = [
      (len(each['table']), name, each['table'])
      for name, each in self.raw_data.items()
    ]
    tables = sorted(tables, key=lambda _:_[0])
    t0 = time.time()
    for length, name, table in tables:
      atable = table['_id', 'ra', 'dec', 'coord']
      if len(self.matched) == 0: # first table, use all data
        for each in atable.iterrows():
          _id,ra,dec,coord = each
          self.matched.add_row({
              'ra':ra,
              'dec':dec,
              'count':1,
              'M:'+name: _id if len(_id) else '*'
            })
      else: # do match
        for each in atable.iterrows():
          _id,ra,dec,coord = each
          matched = None
          dec_over = None
          for index, e in enumerate(self.matched.iterrows()):
            eid, ecount, era, edec, *__ = e
            if edec < dec-radius:
              continue
            if dec_over is None and edec > dec:
              dec_over = index
            if edec > dec+radius:
              break
            ecoord = SkyCoord(ra=era, dec=edec, unit=(u.deg, u.deg))
            sep = coord.separation(ecoord).deg
            if sep < radius:
              matched = index
              break
          if matched is not None:
            self.matched[matched]['M:'+name] = _id
            self.matched[matched]['count'] += 1
          else:
            row = {
              'ra':ra,
              'dec':dec,
              'count':1,
              'M:'+name: _id if len(_id) else '*'
            }
            if dec_over is None:
              self.matched.add_row(row)
            else:
              self.matched.insert_row(dec_over, row)
    self.delta = time.time() - t0
    self._get_id()
  def _get_id(self):
    '''set name to the most one'''
    for i in range(len(self.matched)):
      each = self.matched[i]
      colnames = list(filter(lambda _:_.startswith('M:'),each.colnames))
      names = list(map(lambda _:each[_], colnames))
      names = list(filter(lambda _:_ not in ['', '*'], names))
      names = set(names)
      names = list(names)
      names.sort(key=lambda _:len(_), reverse=True)
      if len(names):
        self.matched[i]['id'] = names[0]
      else:
        coord = SkyCoord(ra=each['ra'],dec=each['dec'],unit=(u.deg, u.deg))
        self.matched[i]['id'] = coord2str(coord)

# functions
def yamldump(*args, **kwargs):
  return yaml.safe_dump(*args, **kwargs, width=200, indent=2)

def save_vars(keys, name, *, globals):
  tosave = {key:globals[key] for key in keys}
  with open(name, 'wb') as f:
    pickle.dump(tosave, f)
def load_vars(name, *, globals):
  with open(name, 'rb') as f:
    toload = pickle.load(f)
  for key in toload:
    globals[key] = toload[key]

class ParallelDo:
  def __init__(self, count=None, path=None):
    if count is None or count is 0:
      self.count = multiprocessing.cpu_count() - 1
    else:
      self.count = count
    if path is None:
      raise Exception('should give log path')
    self.process = {}
    self.l = logger = logging.getLogger('parallel') # return name root
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter.datefmt = "%Y-%m-%dT%H:%M:%S"
    fh = logging.FileHandler(path)
    ch = logging.StreamHandler()
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh); logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.propagate=False
    logger.info("========{}=========".format(datetime.now()))
  def _add_loop(self, func, kwargs, name):
    for key,p in list(self.process.items()):
      if not p.is_alive():
        self.l.info('remove {}'.format(key))
        self.process.pop(key)
    if len(self.process) < self.count:
      self.l.info('   add {}'.format(name))
      self.process[name] = multiprocessing.Process(
        target=self.log_wrapper(func, name), kwargs=kwargs
      )
      self.process[name].start()
      return self.process[name]
    else:
      return None
  def add(self, func, kwargs, name=None):
    if name is None:
      name = str(time.time())
    while True:
      time.sleep(1)
      success = self._add_loop(func, kwargs, name)
      if success is not None:
        break
  def _get_exp(self, e):
    if not hasattr(e, 'recorded'):
      exp = traceback.format_exc()
      ss = exp.split('\n')
      ss = filter(lambda _:_.strip(),ss)
      output = list(map(lambda _:'  '+_, ss))
      e.recorded = True
      return '\n'.join(output)
    else:
      return '  see above'
  def log_wrapper(self, func, name):
    def wrapper(*args, **kwargs):
      try:
        return func(*args, **kwargs)
      except Exception as e:
        self.l.critical(' error in {}:\n{}'.format(name, self._get_exp(e)))
    return wrapper

def DEXEC(cursor, SQL, values):
  N = 0
  while True:
    N += 1
    try:
      return cursor.execute(SQL, values)
      break
    except Exception as e:
      if N>20:
        raise
      time.sleep(np.random.randint(10)+1)
def pick(obj, keys):
  result = {}
  for key in keys:
    if key in obj:
      result[key] = obj[key]
  return result

if 'constants':
  ftype_sort_map = {
    'SX1': 0,
    'X1DSUM': 1,
    'X1D': 2,
    'SX2': 3,
    'X2D': 4,
    'FLT': 5,
    }
