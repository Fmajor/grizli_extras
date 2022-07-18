from pyds9 import DS9
from astropy import wcs as WCS
import astropy.io.fits as fits
import os
import time
import textwrap
import re
import numpy as np
import copy
import tempfile

try:
  import galsim
except Exception as e:
  # try to import galsim
  pass

if 'setters':
  # mode setter
  modeList = ["none", "region", "crosshair",
              "colorbar", "pan", "zoom", "rotate"]
  class ModeSetter(object):
      def __init__(self, father):
          self._father = father
      @property
      def pan(self):
          self._father.set("mode pan")
      @property
      def rotate(self):
          self._father.set("mode rotate")
      @property
      def none(self):
          self._father.set("mode none")
      @property
      def region(self):
          self._father.set("mode region")
      @property
      def zoom(self):
          self._father.set("mode zoom")
      @property
      def crosshair(self):
          self._father.set("mode crosshair")
      @property
      def crop(self):
          self._father.set("mode crop")

  # match setcher
  matchLevel2 = ["wcs", "image", "physical", "amplifier", "detector"]
  submatchStr=textwrap.dedent("""
  @property
  def {match}(self):
      if self._name=="slice" and '{match}' in ["physical", "amplifier", "detector"]:
          print("unchanged")
      else:
          self._father.set("match {{}} {match}".format(self._name))
  """)
  class SubMatchSetter(object):
      def __init__(self, father, name):
          self._father = father
          self._name = name

      for eachSubMatch in matchLevel2:
          exec(submatchStr.format(match=eachSubMatch), globals(), locals())

  matchSetterStr=textwrap.dedent("""
  @property
  def {matchname}(self):
      self._father.set("match {match}")
  """)
  matchLevel1 = ["bin", "scale", "colorbar", "smooth", "scalelimits"]
  matchLevel1Name = ["bin", "scale", "colorbar", "smooth", "limits"]
  matchLevel12 = ["frame", "crosshair", "crop", "slice"]

  class MatchSetter(object):
      def __init__(self, father):
          self._father = father
          for eachSubMatch in matchLevel12:
              exec("self.{f} = SubMatchSetter(self._father,'{f}')".format(f=eachSubMatch), globals(), locals())

      for eachMatch,eachMatchName in zip(matchLevel1, matchLevel1Name):
          todo = matchSetterStr.format(match = eachMatch, matchname=eachMatchName)
          exec(todo, globals(), locals())

      @property
      def limits(self):
          thisLimit = self._father.get("scale limists")
          self._father.set("match scale limits {}".format(thisLimit))
      @limits.setter
      def limits(self, value):
          if len(value)==2:
              try:
                  float(value[0])
                  float(value[1])
                  self.set("match scale limits {} {}".\
                          format(value[0], value[1]))
                  return
              except:
                  raise Exception("invaild value or data type for {}: {}, type {}".\
                      format(sys._getframe().f_code.co_name, value, type(value)))

  # lock secter
  lockLevel2 = ["wcs", "none", "image", "physical", "amplifier", "detector"]
  sublockStr=textwrap.dedent("""
  @property
  def {lock}(self):
      if self._name=="slice" and '{lock}' in ["physical", "amplifier", "detector"]:
          print("unchanged")
      else:
          self._father.set("lock {{}} {lock}".format(self._name))
  """)
  class SubLockSetter(object):
      def __init__(self, father, name):
          self._father = father
          self._name = name

      for eachSubLock in lockLevel2:
          exec(sublockStr.format(lock=eachSubLock), globals(), locals())
      @property
      def get(self):
          return self._father.get("lock {}".format(self._name))

  lockSetterStr=textwrap.dedent("""
  @property
  def {lockname}(self):
      result = self._father.get("lock {lock}")
      if result[0]=="y":
          self._father.set("lock {lock} off")
          return 0
      elif result[0]=="n":
          self._father.set("lock {lock} on")
          return 1
      else:
          print(result)
          raise Exception("unknown result from {lock}")

  """)

  lockSetterStrOn=textwrap.dedent("""
  @property
  def {lockname}_on(self):
      result = self._father.set("lock {lock} on")
  """)
  lockSetterStrOff=textwrap.dedent("""
  @property
  def {lockname}_off(self):
      result = self._father.set("lock {lock} off")
  """)

  lockLevel1     = ["bin", "scale", "colorbar", "smooth", "scalelimits", ]
  lockLevel1Name = ["bin", "scale", "colorbar", "smooth", "scalelimits", ]
  lockLevel12 = ["frame", "crosshair", "crop", "slice"]

  class LockSetter(object):
      def __init__(self, father):
          self._father = father
          for eachSubLock in lockLevel12:
              exec("self.{f} = SubLockSetter(self._father,'{f}')".format(f=eachSubLock), globals(), locals())
      for eachLock, eachLockName in zip(lockLevel1, lockLevel1Name):
          todo = lockSetterStr.format(lock = eachLock, lockname=eachLockName)
          exec(todo, globals(), locals())
      for eachLock, eachLockName in zip(lockLevel1, lockLevel1Name):
          todo = lockSetterStrOn.format(lock = eachLock, lockname=eachLockName)
          exec(todo, globals(), locals())
      for eachLock, eachLockName in zip(lockLevel1, lockLevel1Name):
          todo = lockSetterStrOff.format(lock = eachLock, lockname=eachLockName)
          exec(todo, globals(), locals())
      @property
      def get(self):
          allResult = []
          for eachLock in lockLevel1:
              result = self._father.get("lock {}".format(eachLock))
              if result[0]=="y":
                  allResult.append(eachLock)
              elif result[0]!="n":
                  print(result)
                  raise Exception("unknown result from {}".format(eachLock))
          return allResult
if 'functions':
  def getDimStr(header):
      extName = header.get("extname", "")
      dim = header.get("NAXIS", 0)
      dims = ", ".join([str(header.get("NAXIS"+str(each),"?")) for each in range(1, dim+1)])
      obj = header.get("object", "")
      if not dim:
          return "{} {} {}".format(extName, dim, obj)
      else:
          return "{} {} [{}] {}".format(extName, dim, dims, obj)
if 'regions':
  regionTypes = ["circle", "ellipse", "box", "polygon", "point",
                 "line", "vector", "text", "ruler", "compass",
                 "projection", "annulus", "ellipse", "box", "panda",
                 "epanda", "bpanda", "composite"]

  str_in_parentheses = r"\((.*?)\)"
  re_in_parentheses = re.compile(str_in_parentheses)
  str_before_parentheses = r"(\w*)\("
  re_before_parentheses = re.compile(str_before_parentheses)

  str_tag = r"tag=\{(.*?)\}"
  re_tag = re.compile(str_tag)

  str_inside = r"\{(.*?)\}"
  re_inside = re.compile(str_inside)

  class Region(object):
      def __init__(self):
          pass
          #self.defaultFont = {"fontname":"times", "size":"12", "bold":"normal", "italic":"roman"}
          #self.defaultFontKeys = ["fontname", "size", "bold", "italic"]
      def dict2eq(self, d):# in d, text and tag and fond already inside ''
          tagList = d.get("tag",[])
          result = ",".join(["{}={}".format(eachKey, d[eachKey]) for eachKey in d if eachKey != "tag"])
          resultTag = ",".join(["tag={}".format(eachTag) for eachTag in tagList])
          if resultTag:
              if result:
                  return ",".join([result, resultTag])
              else:
                  return resultTag
          else:
              return result
      def processRegionLine(self,s):
          if "file format" in s:
              return s
          if s in ["image", "physical"]:
              return s
          if s == "wcs;":
              return "wcs"
          if "global" in s:
              command="global"
              params=""
              afterCommand=s
          else:
              aux = s.split("#")
              if "(" in aux[0] and ")" in aux[0]: # command in first part
                  command = re_before_parentheses.findall(aux[0])[0]
                  params = re_in_parentheses.findall(aux[0])[0].split(",")
                  if len(aux)==2:
                      afterCommand = aux[1].strip()
                  else:
                      afterCommand = ""
              elif "(" in aux[1] and ")" in aux[1]:
                  command = re_before_parentheses.findall(aux[1])[0]
                  params = re_in_parentheses.findall(aux[1])[0].split(",")
                  afterCommand = aux[1].split(")")[-1].strip()
              else:
                  raise Exception("string {} not understood".format(s))
          configAux = afterCommand.split("=")
          totalN = len(configAux)-1
          if totalN == 0:
              return command, params, {}
          configPairs = [""] * totalN
          haveTag = False
          for i,each in enumerate(configAux):
              thisPair = each.split(" ")
              #print(i, thisPair)
              N = len(thisPair)
              if i>0:
                  if i<totalN:
                      thisKey = thisPair[-1].strip()
                      if thisKey=="tag":
                          haveTag=True
                      configPairs[i] = [thisKey, "?"]
                      configPairs[i-1][1] = " ".join(thisPair[0:-1])
                  else:
                      configPairs[i-1][1] = each
              else:
                  thisKey = thisPair[0].strip()
                  if thisKey=="tag":
                      haveTag=True
                  configPairs[i] = [thisKey, "?"]
          #print(configPairs)
          if haveTag:
              tagList = [eachPair[1][1:-1] for eachPair in configPairs if eachPair[0]=="tag"]
          configs = dict(configPairs)
          if haveTag:
              configs["tag"] = tagList
          if "font" in configs.keys():
              values = configs["font"][1:-1].split()
              assert len(values)==4
              configs["font"] = {self.defaultFontKeys[i]:eachValue for i,eachValue in enumerate(values)}
          if "text" in configs.keys():
              configs["text"] = configs["text"][1:-1]
          if command=="global":
              return command, configs

          return command, params, configs
      def genRegionCommand(self, command, params, configs={}, sys=None, **kwargs):
          myParams = [str(each) for each in params]
          front = "{command}({params})".format(command=command, params = ",".join(myParams))
          c = {}
          c.update({"tag":"pyds9"})
          c.update(configs)
          if kwargs is not None:
              c.update(kwargs)
          fontKeys = list(set(self.defaultFontKeys).intersection(set(c.keys())))
          if len(fontKeys)>0 or "font" in c.keys():
              if "font" in c.keys():
                  if type(c["font"])==str or type(c["font"])==unicode:
                      thisFontConfig = dict([(eachKey, eachValue)
                          for eachKey, eachValue in
                               zip(self.defaultFontKeys, c["font"].split())])
                  elif type(c["font"])==dict:
                      thisFontConfig = copy.copy(c["font"])
              else:
                  thisFontConfig = copy.copy(self.defaultFont)
              for eachKey in fontKeys:
                  thisFontConfig[eachKey] = c[eachKey]
              #print(c)
              #print(thisFontConfig)
              #print(self.defaultFontKeys)
              thisFontStr = " ".join([thisFontConfig[eachkey] for eachkey in self.defaultFontKeys])
              c["font"] = "'{}'".format(thisFontStr)
              for eachKey in fontKeys:
                  c.pop(eachKey)
          if "tag" in c.keys():
              if type(c["tag"])==str or type(c["tag"])==unicode:
                  aux = ["'{}'".format(each) for each in c["tag"].split(",")]
                  c["tag"] = aux
              elif type(c["tag"])==list or type(c["tag"])==tuple:
                  aux = ["'{}'".format(each) for each in c["tag"]]
                  c["tag"] = aux
              else:
                  raise Exception("known type of tag: {}".format(c["tag"]))
          if "text" in c.keys():
              c["text"] = "'{}'".format(c["text"])
          end = self.dict2eq(c)
          if sys is None:
            sys = self.rsys
          return "regions command \"{sys};{front} # {end}\"".format(
            front=front,
            end=end,
            sys=sys,
            )
      def addRegion(self, command, params, configs={}, **kwargs):
          c = copy.copy(configs)
          c.update(kwargs)
          command = self.genRegionCommand(command, params, c)
          print(command)
          try:
              self.set(command)
          except:
              print("error command:", command)
      def _resampleLine(self, params):
          x1,y1,x2,y2,width = params
          x1 = float(x1); y1 = float(y1);
          x2 = float(x2); y2 = float(y2);
          width = float(width)
          if self.rsys == "wcs":
              w = WCS.WCS(self.header)
              ((x1,y1), (x2,y2))= w.wcs_world2pix([[x1, y1], [x2, y2]], 1)
              print('use wcs: {} ==> {}'.format(params, [x1,y1, x2,y2, width]))
          p1 = np.array([x1,y1], dtype=np.float64)
          p2 = np.array([x2,y2], dtype=np.float64)
          dp = p2-p1
          length = np.sqrt(np.sum((p2-p1)**2))
          dpN = dp/length
          if width<2:
              N = 1
          else:
              N = int(width)
          idpN = np.array([-dpN[1], dpN[0]])
          iLength = int(length)+1
          #step = np.linspace(0, 1, iLength)
          step = np.arange(iLength)
          result = np.zeros((iLength,N), dtype=np.dtype([("x", np.float64), ("y", np.float64)]))
          iresult = np.zeros((iLength,N), dtype=np.dtype([("x", np.int64), ("y", np.int64)]))
          for i in range(N):
              xx = x1 + step * dpN[0]
              yy = y1 + step * dpN[1]
              result["x"][:,i] = xx
              result["y"][:,i] = yy
              iresult["x"][:,i] = np.round(xx)-1
              iresult["y"][:,i] = np.round(yy)-1
              x1 += idpN[0]
              y1 += idpN[1]
          return iresult, result# result[:,0]

class cDS9(DS9, Region):
  def __init__(self, name=None, init=True, **kwargs):
    if name is not None:
      super().__init__(name)
    else:
      super().__init__()

    self.m = ModeSetter(self)
    self.lock = LockSetter(self)
    self.match = MatchSetter(self)

    self.rconfig={"format":"ds9", "system":"image", 'color':'Cyan'}
    self.defaultFont = {"fontname":"times", "size":"12", "bold":"normal", "italic":"roman"}
    self.defaultFontKeys = ["fontname", "size", "bold", "italic"]

    self.frame_tags = {}

    if init:
      #self.set("cmap Heat")
      self.set("lock scale")
      self.set("lock colorbar")
      #self.set("region shape projection")
      #self.tile
      self.zscale
      self.lock.frame.image
      self.nan = "red"
      #self.m.none
      self.rsys = 'image'
      self.rcolor = 'magenta'

    width = kwargs.get("width")
    height = kwargs.get("height")
    if width is not None:
        self.width = width
    if height is not None:
        self.height = height
  #<== only get
  @property
  def frames(self):
      return list(map(int, self.get("frame all").split()))
  @property
  def info(self):
      print("cd: {}".format(self.get("cd")))
      print("file: {}".format(self.get("file")))
      print("frame: {}".format(self.get("frame")))
      print("pan: {} | ".format(self.get("pan")),end="")
      print("rotate: {} | ".format(self.get("rotate")),end="")
      print("zoom: {} {} | ".format(self.get("zoom x"), self.get("zoom y")),end="")
      print("windows: {} {}".format(self.get("width"), self.get("height")))
      print("scale: {} | ".format(self.get("scale")),end="")
      print("bin: {} | ".format(self.get("bin factor")),end="")
      print("crop: {} | ".format(self.get("crop")),end="")
      print("crosshair: {}".format(self.get("crosshair")),)
      print("mode: {}: | ".format(self.get("mode")),end="")
      print("nan: {} | ".format(self.get("nan")),end="")
      print("orient: {} | ".format(self.get("orient")),end="")
  #==>
  #<== set and get
  @property
  def mode(self):
      return self.get("mode")
  @mode.setter
  def mode(self, value):
      self.set("mode {}".format(value))
  @property
  def frame(self):
      return int(self.get("frame"))
  @frame.setter
  def frame(self, value):
      try:
          self.set("frame {}".format(int(value)))
      except Exception as e:
          print(e)
          raise Exception("invaild value or data type for {}: {}, type {}".\
                  format(sys._getframe().f_code.co_name, value, type(value)))
  @property
  def rotate(self):
      return float(self.get("rotate"))
  @rotate.setter
  def rotate(self, value):
      try:
          self.set("rotate {}".format(float(value)-self.rotate))
      except Exception as e:
          print(e)
          raise Exception("invaild value or data type for {}: {}, type {}".\
                  format(sys._getframe().f_code.co_name, value, type(value)))
  @property
  def blink(self):
      result = self.get("blink")
      if result=="yes":
          return 1
      elif result=="no":
          return 0
  @blink.setter
  def blink(self, value):
      assert value, isinstance(value, int) or isinstance(value, float)
      result = self.set("blink interval {}".format(value))
  @property
  def blink_on(self):
      self.set("blink yes")
  @property
  def blink_off(self):
      self.set("blink no")

  @property
  def zoom(self):
      aux = self.get("zoom")
      if aux:
          aux = aux.split()
          if len(aux)==1:
              return float(aux[0])
          else:
              return float(aux[0]), float(aux[1])
      else:
          return None
  @zoom.setter
  def zoom(self, value):
      try:
          if isinstance(value, int) or isinstance(value, float):
              self.set("zoom to {}".format(value))
          else:
              self.set("zoom to {} {}".format(value[0], value[1]))
      except Exception as e:
          print(e)
          raise Exception("invaild value or data type for {}: {}, type {}".\
                  format(sys._getframe().f_code.co_name, value, type(value)))
  @property
  def zoomfit(self):
      self.set("zoom to fit")
  @property
  def bin(self):
      aux = self.get("bin factor")
      if aux:
          aux = aux.split()
          if len(aux)==1:
              return float(aux[0])
          else:
              return float(aux[0]), float(aux[1])
      else:
          return None
  @bin.setter
  def bin(self, value):
      try:
          if isinstance(value, int) or isinstance(value, float):
              self.set("bin factor {}".format(value))
          else:
              self.set("bin factor {} {}".format(value[0], value[1]))
      except Exception as e:
          print(e)
          raise Exception("invaild value or data type for {}: {}, type {}".\
                  format(sys._getframe().f_code.co_name, value, type(value)))
  @property
  def pan(self):
      temp = self.get("pan").split()
      return np.array([float(each) for each in temp])
  @pan.setter
  def pan(self, value):
      current = self.pan
      if len(value)==2:
          self.set("pan {} {}".\
                  format(float(value[0]), float(value[1])))
      else:
          raise Exception("invaild value or data type for {}: {}, type {}".\
              format(sys._getframe().f_code.co_name, value, type(value)))
  @property
  def panto(self):
      temp = self.get("pan").split()
      return np.array([float(each) for each in temp])
  @pan.setter
  def panto(self, value):
      current = self.pan
      if len(value)==2:
          try:
              self.set("pan {} {}".\
                      format(float(value[0])-current[0], float(value[1])-current[1]))
          except:
              raise Exception("invaild value or data type for {}: {}, type {}".\
                  format(sys._getframe().f_code.co_name, value, type(value)))
      elif len(value)==3:
              self.set("pan to {} {} {}".format(float(value[0]),
                                             float(value[1]), value[2]))
      elif len(value)==4:
              self.set("pan to {} {} {} {}".format(float(value[0]),
                                                float(value[1]), value[2], value[3]))
  @property
  def nan(self):
      return self.get("nan")
  @nan.setter
  def nan(self, value):
      self.set("nan {}".format(value))
  @property
  def limits(self):
      temp = self.get("scale limits").split()
      return np.array([float(each) for each in temp])
  @limits.setter
  def limits(self, value):
      current = [int(each) for each in self.limits]
      if len(value)==2:
          try:
              float(value[0])
              float(value[1])
              self.set("scale limits {} {}".\
                      format(value[0], value[1]))
              return
          except:
              raise Exception("invaild value or data type for {}: {}, type {}".\
                  format(sys._getframe().f_code.co_name, value, type(value)))
  @property
  def scale(self):
      return self.get("scale")
  @scale.setter
  def scale(self, value):
      valueList = ["linear","log","pow","sqrt","squared","asinh","sinh","histequ"]
      if value in valueList:
          self.set("scale {}".format(value))
          return
      raise Exception("invaild value or data type for {}: {}, type {}\n\tvalue must be a string in {}".\
              format(sys._getframe().f_code.co_name, value, type(value), valueList))
  @property
  def zc(self):
      return float(self.get('zscale contrast'))
  @zc.setter
  def zc(self, value):
      self.set('zscale contrast {}'.format(value))
  @property
  def cropnone(self):
      x,y = self.shape
      self.cropbox=1,1,x,y
  @property
  def crop(self):
      result = self.get("crop").split()
      return [float(each) for each in result]
  @crop.setter
  def crop(self, value):
      if value is None:
          self.set("crop reset")
          return
      elif len(value) == 4:
          try:
              int(value[0]);float(value[0])
              int(value[1]);float(value[1])
              int(value[2]);float(value[2])
              int(value[3]);float(value[3])
              self.set("crop {} {} {} {}".format(value[0], value[1], value[2], value[3]))
              return
          except Exception as e:
              print(e)
      raise Exception("invaild value or data type for {}: {}, type {}".\
              format(sys._getframe().f_code.co_name, value, type(value)))
  @property
  def cropbox(self):
      result = self.get("crop").split()
      result = list(map(float, result))
      x, y, dx, dy = result
      return [x-(dx-1)/2, y-(dy-1)/2, x+(dx-1)/2, y+(dy-1)/2]
  @cropbox.setter
  def cropbox(self, value):
      if value is None:
          self.set("crop reset")
          return
      elif len(value) == 4:
          try:
              int(value[0]);float(value[0])
              int(value[1]);float(value[1])
              int(value[2]);float(value[2])
              int(value[3]);float(value[3])
              #self.set("crop {} {} {} {}".format(value[0], value[1], value[2], value[3]))
              # x1, y1, x2, y2
              # y is the first index in numpy
              self.set("crop {} {} {} {}".
                          format( (value[2]+value[0])/2, (value[3]+value[1])/2,
                                  (value[2]-value[0]+1), (value[3]-value[1]+1) ))
              return
          except Exception as e:
              print(e)
      raise Exception("invaild value or data type for {}: {}, type {}".\
              format(sys._getframe().f_code.co_name, value, type(value)))
  @property
  def width(self):
      return float(self.get("width"))
  @width.setter
  def width(self, value):
      assert value, isinstance(value, int) or isinstance(value, float)
      self.set("width {}".format(value))
  @property
  def height(self):
      return float(self.get("height"))
  @height.setter
  def height(self, value):
      assert value, isinstance(value, int) or isinstance(value, float)
      self.set("height {}".format(value))
  @property
  def window(self):
      height = self.height
      width = self.width
      return width, height
  @window.setter
  def window(self, value):
      assert type(value) in [list, tuple]
      if len(value)==2:
          try:
              self.width = int(value[0])
              self.height = int(value[1])
              return
          except Exception as e:
              print(e)
      raise Exception("invaild value or data type for {}: {}, type {}".\
                  format(sys._getframe().f_code.co_name, value, type(value)))
  @property
  def wcs_format(self):
      return self.get('wcs format')
  @wcs_format.setter
  def wcs_format(self, value):
      if value not in ['degrees', 'sexagesimal']:
          raise ValueError('wcs format can only be degrees or sexagesimal')
      else:
          self.set('wcs format {}'.format(value))

  #==> set and get
  #<== only set
  @property
  def top(self):
      self.set("raise")
  @property
  def single(self):
      self.set("single")
  @property
  def tile(self):
      self.set("tile")
  @property
  def zscale(self):
      self.set("zscale")
  @property
  def minmax(self):
      self.set("minmax")
  @property
  def fclean(self):
    self.frame_tags = {}
    self.set("frame delete all")
  #==> only set
  #<== about data
  @property
  def shape(self):
      "return shape of (x, y), note that the readin pyfits data is of shape (y,x)"
      data = self.get_arr2np()
      return data.T.shape
  @property
  def data(self):
      self._thisData = self.pyfits.data
      return self._thisData
  @property
  def pyfits(self):
      self._thisPyfits = self.get_fits()[0]
      return self._thisPyfits
  @property
  def header(self):
      self._thisHeader=self.get_fits()[0].header
      return self._thisHeader
  def open(self, name, frames=None):
      "open a fits file in new frames"

      try:#!! test if there has no data frame yet
          data = self.data
      except:
          self.set("frame delete")
      _frames = self.frames
      __ = [int(each) for each in _frames]
      if len(__)==0:
          startNumber = 1
          print("add new frame: {}".format(startNumber))
          self.frame = startNumber
      else:
          startNumber = np.array(__).max()+1

      _frames = []
      self._data = fits.open(name)

      if frames is not None:
        todo = frames
      else:
        N = len(self._data)
        todo = range(N)
      for i in todo:
        if startNumber != self.frame:
          print("add new frame: {}".format(startNumber))
          self.frame = startNumber
        try:
          thisName = "{}[{}]".format(name, i)
          print("\t", thisName, "", getDimStr(fits.open(name)[i].header), end='')
          self.set("fits \"{}\"".format(thisName))
          print()
          _frames.append(startNumber)
          startNumber += 1
        except:
          print('  ERROR')
      return _frames
  @property
  def newFrame(self):
      try:
          data = self.data
      except:
          self.set("frame delete")
      frames = self.frames
      aux = [int(each) for each in frames]
      if len(aux)==0:
          startNumber = 1
      else:
          startNumber = np.array(aux).max()+1
      print("add new frame: {}".format(startNumber))
      self.frame = startNumber
      return startNumber
  @property
  def xy(self):
      result = self.get("iexam coordinate image").split()
      return [float(result[0]), float(result[1])]
  @property
  def kxy(self):
      result = self.get("iexam key").split()
      return [result[0], float(result[1]), float(result[2])]
  def _to_showable(self, data):
    if isinstance(data, (list, tuple)):
      lists = [self._to_showable(_) for _ in data]
      l = flat_list = [item for sublist in lists for item in sublist]
    elif isinstance(data, np.ndarray):
      l = [data]
    elif isinstance(data, (
      fits.hdu.hdulist.HDUList,
      fits.hdu.image.PrimaryHDU,
      fits.hdu.image.ImageHDU,
      )):
      if not isinstance(data, fits.hdu.hdulist.HDUList):
        each = data
        l = [
              fits.HDUList([
                fits.PrimaryHDU(
                  each.data, header=each.header
                )
              ])
            ]
      else:
        l = []
        for each in data:
          if each.data is not None and len(each.data.shape)>1:
            l.append(
              fits.HDUList([
                fits.PrimaryHDU(
                  each.data, header=each.header
                )
              ])
            )
    elif 'galsim' in globals() and isinstance(data, galsim.Image):
      hdulist = fits.HDUList()
      data.write(hdu_list=hdulist)
      l = [hdulist]
    else:
        raise Exception("error format: {}".format(type(l)))
    return l
  def show(self, data, tags=None):
      frames = []
      l = self._to_showable(data)
      if tags is not None:
        if len(l) != len(tags):
          raise Exception('length of shown and tags not matched: {}!={}'.format(len(l), len(tags)))
      self._show_list = l
      for i, eachArray in enumerate(l):
        if tags is not None:
          tag = tags[i]
          if tag in self.frame_tags:
            self.frame = self.frame_tags[tag]
            print("find exist frame: {}".format(self.frame_tags[tag]))
            continue
        try:#!! test if there has no data frame yet
            data = self.data
        except:
            self.set("frame delete")
        frames = self.frames
        aux = [int(each) for each in frames]
        if len(aux)==0:
            startNumber = 1
        else:
            startNumber = np.array(aux).max()+1
        print("add new frame: {}".format(startNumber))
        self.frame = startNumber
        if tags is not None:
          tag = tags[i]
          self.frame_tags[tag] = startNumber
        frames.append(startNumber)
        if isinstance(eachArray, np.ndarray):
          self.set_np2arr(eachArray)
        else:
          self.set_fits(eachArray)
      return frames
  def ndarrayList(self, l):
      frames = []
      if isinstance(l, list):
          for eachArray in l:
              try:#!! test if there has no data frame yet
                  data = self.data
              except:
                  self.set("frame delete")
              frames = self.frames
              aux = [int(each) for each in frames]
              if len(aux)==0:
                  startNumber = 1
              else:
                  startNumber = np.array(aux).max()+1
              print("add new frame: {}".format(startNumber))
              self.frame = startNumber
              frames.append(startNumber)
              self.set_np2arr(eachArray)
      else:
          raise Exception("please input a list of ndarray")
      return frames
  #==> about data
  #<== about region
  @property
  def rsys(self):
      return self.rconfig["system"]
  @rsys.setter
  def rsys(self, value):
      if value not in ['wcs', 'image', 'physical']:
          raise ValueError('region system should be wcs, image or physical')
      else:
          self.rconfig["system"] = value
          self.set('regions system {}'.format(value))
  @property
  def rcolor(self):
      return self.rconfig["color"]
  @rcolor.setter
  def rcolor(self, value):
      clist = ['black', 'white', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
      if value not in clist:
          raise ValueError('color should be in {}'.format(clist))
      self.rconfig["color"] = value

  @property
  def rformat(self):
      return self.get('regions format')
  def load_region_str(self, str, all=False):
    f = tempfile.NamedTemporaryFile(prefix='ds9', delete=False)
    f.write(str.encode())
    f.close()
    self.load_region(f.name, all=all)
  def load_region(self, file=None, all=False):
    if file is not None:
      abspath = os.path.abspath(file)
      if not os.path.exists(abspath):
        raise Exception('region file not exists: {}'.format(abspath))
    else: # try to load the region file with the same name
      filename = self.get('file')
      filename = '['.join(filename.split('[')[:-1])
      filename = '.'.join(filename.split('.')[:-1]) # remove .fits
      rfile = filename+'.reg'
      if os.path.exists(rfile):
        abspath = os.path.abspath(rfile)
      else:
        raise Exception('region file {} => {} not exists'.format(self.get('file'), rfile))
    if all:
      self.set("regions load all '{}'".format(abspath))
    else:
      self.set("regions load '{}'".format(abspath))
  @property
  def region(self):
      'information about all regions'
      self.set("regions format " + self.rconfig["format"])
      self.set("regions system " + self.rconfig["system"])
      self._thisRegion = self.get("regions list").split("\n")
      temp = [each for each in self._thisRegion if each]
      self._thisRegion = temp
      self._thisTags = set()
      self._thisTagCount = {}
      self._thisTagObjs = {}
      for each in self._thisRegion:
          if "tag" in each:
              temp = re_tag.findall(each)
              for eachResult in temp:
                  if eachResult in self._thisTags:
                      self._thisTagCount[eachResult] += 1
                      self._thisTagObjs[eachResult].append(each)
                  else:
                      self._thisTags.add(eachResult)
                      self._thisTagCount[eachResult] = 1
                      self._thisTagObjs[eachResult] = [each]
      self._thisUniqueTags = [each for each in self._thisTags
                                  if self._thisTagCount[each]==1]
      return self._thisRegion
  @property
  def regions(self):
      'region info in dict'
      self.set("regions format " + self.rconfig["format"])
      self.set("regions system " + self.rconfig["system"])
      self._thisRegions = {}
      for eachType in regionTypes:
          self._thisRegions[eachType] = []
      self._thisRegions["other"] = []
      temp = [each for each in self.get("regions").split("\n") if each]
      for eachTerm in temp:
          flag = False
          for eachType in regionTypes:
              if (eachType+"(") in eachTerm.lower():
                  self._thisRegions[eachType].append(eachTerm)
                  flag = True
                  break
          if not flag:
              self._thisRegions["other"].append(eachTerm)
      temp=[]
      for eachTerm in self._thisRegions["panda"]:
          if "epanda" in eachTerm:
              self._thisRegions["epanda"].append(eachTerm)
          else:
              temp.append(eachTerm)
      self._thisRegions["panda"] = temp
      for eachKey in list(self._thisRegions.keys()):
          if not self._thisRegions[eachKey]:
              self._thisRegions.pop(eachKey)
          else:
              self._thisRegions[eachKey] = \
                      [self.processRegionLine(eachLine)
                              for eachLine in self._thisRegions[eachKey]]

      return self._thisRegions
  @property
  def sregion(self):
      'information about selected regions'
      self.set("regions format " + self.rconfig["format"])
      self.set("regions system " + self.rconfig["system"])
      self._thisRegion = self.get("regions selected").split("\n")
      temp = [each for each in self._thisRegion if each]
      self._thisRegion = temp
      return self._thisRegion
  @property
  def sregions(self):
      self.set("regions format " + self.rconfig["format"])
      self.set("regions system " + self.rconfig["system"])
      self._thisRegions = {}
      for eachType in regionTypes:
          self._thisRegions[eachType] = []
      self._thisRegions["other"] = []
      temp = [each for each in self.get("regions selected").split("\n") if each]
      for eachTerm in temp:
          flag = False
          for eachType in regionTypes:
              if (eachType+"(") in eachTerm.lower():
                  self._thisRegions[eachType].append(eachTerm)
                  flag = True
                  break
          if not flag:
              self._thisRegions["other"].append(eachTerm)
      temp=[]
      for eachTerm in self._thisRegions["panda"]:
          if "epanda" in eachTerm:
              self._thisRegions["epanda"].append(eachTerm)
          else:
              temp.append(eachTerm)
      self._thisRegions["panda"] = temp
      for eachKey in list(self._thisRegions.keys()):
          if not self._thisRegions[eachKey]:
              self._thisRegions.pop(eachKey)
          else:
              self._thisRegions[eachKey] = \
                      [self.processRegionLine(eachLine)
                              for eachLine in self._thisRegions[eachKey]]
      return self._thisRegions
  @property
  def cleanRegion(self):
      self.set("regions deleteall")
  def addProjection(self, params, configs={}, sys=None, **kwargs):
      "addProjection((y1,x1, y2,x2, width), tag=[], text='')"
      assert len(params)==5
      c = {}; c.update(configs)
      c.update(kwargs)
      command = self.genRegionCommand("projection", params, configs=c, sys=sys)
      try:
          self.set(command)
      except:
          print("error command:", command)
  def addCircle(self, params, configs={}, sys=None, **kwargs):
      "addCircle((x,y, r), tag=[], text='')"
      assert len(params)==3
      c = {}; c.update(configs)
      c.update(kwargs)
      command = self.genRegionCommand("circle", params, configs=c, sys=sys)
      try:
          self.set(command)
      except:
          print("error command:", command)
  def addText(self, xyText, configs={}, sys=None, **kwargs):
      "addText((x,y,'text'), tag=[])"
      assert len(xyText)==3
      c = {}; c.update(configs)
      c.update(kwargs)
      c.update({"text":xyText[2]})
      command = self.genRegionCommand("text", xyText[:2], configs=c, sys=sys)
      try:
          self.set(command)
      except:
          print("error command:", command)
  def addPoint(self, params, configs={"point":"X"}, sys=None, **kwargs):
      "addPoint((x,y), point='X')"
      assert len(params)==2
      c = {}; c.update(configs)
      c.update(kwargs)
      command = self.genRegionCommand("point", params, configs=c, sys=sys)
      try:
          self.set(command)
      except:
          print("error command:", command)
  def plotBox(self, x, y, dx, dy, color=None, do=True, all=False):
    extras = ""
    if color is not None:
      extras = '#'
      colorstr = ' color={}'.format(color)
    else:
      colorstr = ''

    BOX = 'box({x},{y},{dx},{dy}) {extras}{colorstr}'.format(x=x,y=y,dx=dx,dy=dy,extras=extras,colorstr=colorstr)
    if do:
      self.load_region_str(BOX, all=all)
    else:
      return BOX
  def plotText(self, x, y, text, color=None, do=True, all=False):
    if color is not None:
      colorstr = ' color={}'.format(color)
    else:
      colorstr = ''
    TEXT = '# text({x},{y}){colorstr} text={{{text}}}'.format(x=x,y=y,text=text,colorstr=colorstr)
    if do:
      self.load_region_str(TEXT, all=all)
    else:
      return TEXT
  def scatterX(self, xx, yy, color='green', delta=1, do=True):
    POINT  = 'point({x},{y}) #point=x color={color}'
    goodmask = np.isfinite(yy)
    xx = xx[goodmask]
    yy = yy[goodmask]
    xx = xx[::delta]
    yy = yy[::delta]
    N  = len(xx)
    regions = []
    for i in range(N):
      x = xx[i]
      y = yy[i]
      regions.append(POINT.format(x=x,y=y, color=color))
    result = '\n'.join(regions)
    if do:
      self.load_region_str(result)
    else:
      return result
  def scatter(self, xx, yy, delta=1, color='green', s=0.6, type='x', do=True):
    LINE   = 'line({x0},{y0},{x1},{y1}) # line=0 0 color={color}'
    goodmask = np.isfinite(yy)
    xx = xx[goodmask]
    yy = yy[goodmask]
    xx = xx[::delta]
    yy = yy[::delta]
    N  = len(xx)
    regions = []
    s = s/2
    for i in range(N):
      x = xx[i]
      y = yy[i]
      if type=='x':
        regions.append(LINE.format(x0=x-s, x1=x+s, y0=y-s, y1=y+s, color=color))
        regions.append(LINE.format(x0=x-s, x1=x+s, y0=y+s, y1=y-s, color=color))
      elif type=='-':
        regions.append(LINE.format(x0=x-s, x1=x+s, y0=y, y1=y, color=color))
      elif type=='|':
        regions.append(LINE.format(x0=x, x1=x, y0=y-x, y1=y+s, color=color))
    result = '\n'.join(regions)
    if do:
      self.load_region_str(result)
    else:
      return result

  def plotLine(self, xx, yy, color='green', delta=1, width=1):
    regions = []
    LINE   = 'line({x0},{y0},{x1},{y1}) # line=0 0 color={color} width={width}'
    goodmask = np.isfinite(yy)
    xx = xx[goodmask]
    yy = yy[goodmask]
    xx = xx[::delta]
    yy = yy[::delta]
    N = int(len(xx)/2)
    for i in range(N):
      start = i*2
      end   = i*2+1
      x0, x1 = xx[start], xx[end]
      y0, y1 = yy[start], yy[end]
      regions.append(LINE.format(x0=x0, x1=x1, y0=y0, y1=y1, color=color, width=width))
    result = '\n'.join(regions)
    self.load_region_str(result)
  #==> about region

if __name__ == "__main__": # test block
  d = cDS9('haha', init=True)
  data = fits.open('/Users/wujinnnnn/github/fmajorAstroTools/fmajorAstroTools/math/test/map_temp2.fits')
