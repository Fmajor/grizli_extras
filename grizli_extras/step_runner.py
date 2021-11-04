import os
import dis
import sys
import time
import inspect
import logging
import traceback
import json
import textwrap
import requests
import tempfile
import hashlib
import uuid
from bs4 import BeautifulSoup

import nbformat
import nbformat as nbf
import nbconvert
from nbconvert import HTMLExporter
from traitlets.config import Config

## failed trial to share locals() between steps, just leave it here
#class ShareLocals(type):
#  def __new__(cls, name, bases, attrs):
#    attrs['_locals'] = {}
#    for name, func in attrs.items():
#      if not name.startswith('_'):
#        attrs[name] = cls.share_locals(func)
#    return type.__new__(cls, name, bases, attrs)
#  def share_locals(func):
#    def __wrapper__(self, *args, **kwargs):
#      return func(self, *args, **kwargs)
#    return __wrapper__

class Tee(object):
  '''from https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file'''
  def __init__(self, name, mode='a', tee=True, unbuffered=False, indent="  "):
    self.file = open(name, mode)
    self.stdout = sys.stdout
    sys.stdout = self
    self.tee = tee
    self.unbuffered = unbuffered
    self.indent = indent
  def __del__(self):
    import sys
    sys.stdout = self.stdout
    self.file.close()
  def write(self, data, from_logger=False):
    if self.tee:
      self.stdout.write(data)
    end_with_newline = data.endswith('\n')
    if end_with_newline:
      data = data[:-1]
    data = data.split('\n')
    data = list(map(lambda _:self.indent+_ if _ else _, data))
    data = '\n'.join(data)
    if end_with_newline:
      data = data+'\n'
    if 'FROM_LOGGER' in globals():
      return
    self.file.write(data)
    if self.unbuffered:
      self.file.flush()
  def flush(self):
    self.file.flush()

def linenumber_of_member(m):
    try:
      return m[1].run.__code__.co_firstlineno
    except AttributeError:
      return -1

class StreamHandler(logging.StreamHandler):
  def emit(self, record):
      """
      Emit a record.

      If a formatter is specified, it is used to format the record.
      The record is then written to the stream with a trailing newline.  If
      exception information is present, it is formatted using
      traceback.print_exception and appended to the stream.  If the stream
      has an 'encoding' attribute, it is used to determine how to do the
      output to the stream.
      """
      try:
        msg = self.format(record)
        stream = self.stream
        # issue 35046: merged two stream.writes into one.
        FROM_LOGGER = True
        stream.write(msg + self.terminator)
        self.flush()
      except RecursionError:  # See issue 36272
        raise
      except Exception:
        self.handleError(record)


class StepRunning(object):
  def __init__(self,
      configs=None,
      log_root='logs',
    ):
    self.tasks = inspect.getmembers(self)
    self.tasks.sort(key=linenumber_of_member)
    self.tasks = list(filter(lambda _:isinstance(_[1], type) and not _[0].startswith('_'), self.tasks))
    self.tasks = list(map(lambda _:_[0], self.tasks))
    self.log_root = log_root
    if configs is not None:
      self.configs = {}

    os.makedirs(self.log_root, exist_ok=True)

    logger = logging.getLogger(name='StepRunning') # init a logger
    logger.setLevel(logging.DEBUG) # set logger level
    formatter = logging.Formatter('%(message)s')
    #formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s %(message)s')

    self.log_info_file  = os.path.join(self.log_root, 'info.log')
    self.log_debug_file = os.path.join(self.log_root, 'debug.log')

    ch = logging.StreamHandler()
    fh = logging.FileHandler(self.log_info_file)
    dfh = logging.FileHandler(self.log_debug_file)

    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.INFO)
    dfh.setLevel(logging.DEBUG)

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    dfh.setFormatter(formatter)

    logger.addHandler(dfh)
    logger.addHandler(fh)
    logger.addHandler(ch)

    self.l = logger
    self.locals = {}
    self.skip_functions = {}
    if not hasattr(self, 'only'):
      self.only = []
    if not hasattr(self, 'skip'):
      self.skip = []
  def _run(self, rerun=False, **kwargs):
    self.tee = Tee(self.log_debug_file, unbuffered=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time()))
    self.l.info('==========={}==========='.format(now))
    running_tasks = list(filter(lambda _:not _.startswith("skip_"), self.tasks))
    error_flag = False
    for i, name in enumerate(running_tasks):
      task = getattr(self, name)
      try:
        if not rerun:
          if len(self.only):
            skip = not (name in self.only)
          else:
            if name in self.skip:
              skip = True
            else:
              if hasattr(task, 'skip'):
                skip = getattr(task, 'skip')(self)
              else:
                skip = False
        else:
          skip = False
        if skip:
          self.l.info('{:03d} skiping: {}'.format(i, name))
          if hasattr(task, 'always'):
            getattr(task, 'always')(self)
          continue
        else:
          self.l.info('{:03d} running: {}'.format(i, name))
          getattr(task, 'run')(self)
          if hasattr(task, 'always'):
            getattr(task, 'always')(self)
      except Exception as e:
        exp = traceback.format_exc()
        ss = exp.split('\n')
        ss = filter(lambda _:_.strip(), ss)
        output = list(map(lambda _:'  '+_, ss))
        output = '\n'.join(output)
        self.l.error('  error:\n{}'.format(output))
        self.l.error('unfinished steps: {}'.format(running_tasks[i:]))
        error_flag = True
        break
    try:
      if hasattr(self, 'always'):
        getattr(self, 'always')()
    except Exception as e:
      exp = traceback.format_exc()
      ss = exp.split('\n')
      ss = filter(lambda _:_.strip(), ss)
      output = list(map(lambda _:'  '+_, ss))
      output = '\n'.join(output)
      self.l.error('  always error:\n{}'.format(output))
    if not error_flag:
      self.l.info('done')
    self.tee.__del__()

def create_nb():
  nb = nbf.v4.new_notebook()

class CustomPreprocessor(nbconvert.preprocessors.Preprocessor):
  def preprocess(self, nb, resources):
    mycss = textwrap.dedent("""
      .jp-InputPrompt:hover {
        background: #f5f5f5;
      }
      .jp-OutputPrompt:hover {
        background: #f5f5f5;
      }

      .jp-InputPrompt {
        cursor: n-resize !important;
      }
      .hide-input .jp-InputPrompt {
        cursor: s-resize!important;
      }
      .hide-input .jp-InputArea-editor >* {
        display: none;
      }
      .hide-input .jp-InputArea-editor::before {
        content: "•••";
        color: gray;
      }

      .hide-input .jp-RenderedMarkdown >* {
        display: none;
      }
      .hide-input .jp-RenderedMarkdown::before {
        content: "•••";
        color: gray;
      }

      .jp-OutputPrompt {
        cursor: n-resize !important;
      }
      .hide-output .jp-OutputPrompt {
        cursor: s-resize !important;
      }
      .hide-output .jp-OutputArea-output >* {
        display: none;
      }
      .hide-output .jp-OutputArea-output::before {
        content: "•••";
        color: gray;
      }

      .hide-input .jp-InputPrompt::before{
        content: "▼";
      }
      .hide-input .jp-InputPrompt{
        padding-top: 0px;
        padding-bottom: 0px;
      }
      .hide-output .jp-OutputPrompt::before {
        content: "▼";
      }
      .hide-output .jp-OutputPrompt {
        padding-top: 0px;
        padding-bottom: 0px;
      }
      .CodeMirror pre {
        margin: 0px;
        z-index: 0;
      }

      #nb-tools {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
      }
      .nb-tool-button {
        font-size: 48px;
        cursor: ns-resize;
      }
      .nb-tool-container {
        display: flex;
        flex-direction: column;
      }
      .scrolled {
        max-height: 250px;
        scroll-behavior: auto;
        background: #f5f5f5;
      }
    """)
    resources['custom_css'] = mycss
    myscript = textwrap.dedent("""
      function toggle_input (event) {
        let el = event.target
        let p = el.parentElement.classList
        if (p.contains('hide-input')) {
          p.remove('hide-input')
        } else {
          p.add('hide-input')
        }
      }
      function toggle_output (event) {
        let el = event.target
        let p = el.parentElement.classList
        if (p.contains('hide-output')) {
          p.remove('hide-output')
        } else {
          p.add('hide-output')
        }
      }
      function foldall(el) {
        minput  = /^#.*(foldinput)/
        moutput = /^#.*(foldoutput)/
        var cells = document.getElementsByClassName('jp-Cell')
        for (var cell of cells) {
          var input_line = cell.querySelector('.jp-Cell-inputWrapper .CodeMirror pre')
          if (!input_line) continue
          var output_line = cell.querySelector('.jp-Cell-inputWrapper .CodeMirror pre')
          if (!minput.test(input_line.textContent) && !moutput.test(input_line.textContent)) continue

          var input_area  = cell.querySelector('.jp-InputArea')
          var output_areas = cell.querySelectorAll('.jp-OutputArea-child')

          if (minput.test(input_line.textContent)) {
            input_area.classList.add('hide-input')
          }
          if (moutput.test(input_line.textContent)) {
            for (let output_area of output_areas) {
              output_area.classList.add('hide-output')
            }
          }
        }
      }
      function unfoldall(el) {
        var cells = document.getElementsByClassName('jp-Cell')
        for (var cell of cells) {
          var input_area  = cell.querySelector('.jp-InputArea')
          var output_areas = cell.querySelectorAll('.jp-OutputArea-child')

          if (input_area) {
            input_area.classList.remove('hide-input')
          }
          for (let output_area of output_areas) {
            output_area.classList.remove('hide-output')
          }
        }
      }
      function toggle_nb_tool (event) {
        event.preventDefault()
        event.stopPropagation()
        let tool = document.querySelector('.nb-tool-container')
        if (tool.style.display) {
          tool.style.display = ''
        } else {
          tool.style.display = 'none'
        }
      }
      function dbl_toggle_output (event) {
        event.preventDefault()
        event.stopPropagation()
        let el = event.target
        console.log(event)
        while (!el.classList.contains('jp-OutputArea-child')) {
          console.log(el)
          if (!el.parentElement) {
            return
          } else {
            el = el.parentElement
          }
        }
        let p = el.classList
        if (p.contains('hide-output')) {
          p.remove('hide-output')
        } else {
          p.add('hide-output')
        }
      }
      function dbl_toggle_output (event) {
        event.preventDefault()
        event.stopPropagation()
        let el = event.target
        console.log(event)
        while (!el.classList.contains('jp-OutputArea-output')) {
          if (!el.parentElement) {
            return
          } else {
            el = el.parentElement
          }
        }
        let p = el.classList
        if (p.contains('scrolled')) {
          p.remove('scrolled')
        } else {
          p.add('scrolled')
        }
      }
      setTimeout(() => {
        for (let el of document.getElementsByClassName('jp-InputPrompt')) {
          el.onclick=toggle_input
        }
        for (let el of document.getElementsByClassName('jp-OutputPrompt')) {
          el.onclick=toggle_output
        }
        for (let el of document.getElementsByClassName('jp-OutputArea-child')) {
          el.ondblclick=dbl_toggle_output
        }
        let button = document.querySelector('.nb-tool-button')
        button.onclick = toggle_nb_tool
      }, 2000)
    """)
    resources['custom_script'] = myscript

    mydiv = textwrap.dedent("""
      <div id="nb-tools">
        <div class="nb-tool-button">
          ⚙
        </div>
        <div class="nb-tool-container" style="display:none;">
          <button id="autofold" onclick="foldall(document.getElementById('autofold'))">
            收
          </button>
          <div style="height:5px;"></div>
          <button id="autounfold" onclick="unfoldall(document.getElementById('autounfold'))">
            开
          </button>
        </div>
      </div>
    """)
    resources['custom_div'] = mydiv
    return nb, resources

class MyHTMLExporter(HTMLExporter):
  def __init__(self, *args, cache='/tmp/nbconvert', **kwargs):
    self.extra_template_basedirs.append(
      os.path.abspath(
        os.path.join(
          os.path.dirname(__file__), 'jupyter/templates'
        )))
    super().__init__(*args, **kwargs)
    self.cache = cache
    os.makedirs(self.cache, exist_ok=True)
  def from_notebook_node(self, nb):
    body, resources = super().from_notebook_node(nb)
    html = BeautifulSoup(body, 'html.parser')
    md5 = hashlib.md5()
    # embed the remote scripts
    for script in html.find_all('script'):
      if script.get('id') == 'lab-custom-folding':
        script.decompose()
        continue
      if len(script.contents):
        if "document.getElementById('autofold').click()" in script.contents[0]:
          script.decompose()
          continue
      src_url = script.get('src')
      if src_url:
        if 'mathjax' in src_url:
          continue
        md5.update(src_url.encode('utf-8'))
        shash = md5.hexdigest()
        tfile = os.path.join(self.cache, shash+'.js')
        if not os.path.exists(tfile):
          src = requests.get(src_url).text
          with open(tfile, 'w') as f:
            f.write(src)
        else:
          with open(tfile) as f:
            src = f.read()
        del script['src']
        script['hash'] = shash
        script.string = src

    # set folds
    cells = html.find_all(lambda _:_.has_attr('cell_id'))
    css = html.find(id='custom_style')
    for hcell in cells:
      id = hcell['cell_id']
      type = hcell['cell_type']
      for cell in nb['cells']:
        if cell.get('id') == id:
          break
      else:
        continue
      if cell.get('metadata',{}).get('jupyter',{}).get('source_hidden'):
        el = hcell.find(lambda _:'jp-InputArea' in _.get('class', {}))
        el['class'].append('hide-input')
        hide_str = cell.get('metadata',{}).get('jupyter',{}).get('source_hidden_str')
        if hide_str:
          new_css = textwrap.dedent("""
          [cell_id="{id}"] .hide-input .jp-InputArea-editor::before {{
            content: "••• {str}";
          }}
          """.format(id=id, str=hide_str))
          css.string = css.string + new_css
      if cell.get('metadata',{}).get('jupyter',{}).get('outputs_hidden'):
        els = hcell.find_all(lambda _:'jp-OutputArea-child' in _.get('class',{}))
        for el in els:
          el['class'].append('hide-output')
        hide_str = cell.get('metadata',{}).get('jupyter',{}).get('outputs_hidden_str')
        if hide_str:
          new_css = textwrap.dedent("""
          [cell_id="{id}"] .hide-output .jp-OutputArea-output::before {{
            content: "••• {str}";
          }}
          """.format(id=id, str=hide_str))
          css.string = css.string + new_css

      if cell.get('metadata', {}).get('scrolled'):
        els = hcell.find_all(lambda _:'jp-OutputArea-output' in _.get('class',{}))
        for el in els:
          el['class'].append('scrolled')
    body = str(html)
    return body, resources

class NB(object):
  @classmethod
  def load(cls, file):
    with open(file) as f:
      nb_json = nbformat.read(f, 4)
    nb = cls()
    nb.nb = nb_json
    nb.file = file
    return nb
  def __init__(self, file=None):
    self.file = file
    if file is not None:
      if os.path.exists(file):
        with open(file) as f:
          self.nb = nbformat.read(f, 4)
      else:
        self.nb = nbf.v4.new_notebook()
    else:
      self.nb = nbf.v4.new_notebook()
  def add_code(self, code='', *, name=None, collapsed=None, scrolled=None, hide_source=None, hide_output=None):
    cell = nbf.v4.new_code_cell(code)
    if collapsed is not None:
      cell['metadata']['collapsed'] = collapsed
    if scrolled is not None:
      cell['metadata']['scrolled'] = scrolled
    if name is not None:
      cell['_name'] = name
    if hide_source or hide_output:
      cell['metadata'] = {'jupyter':{}}
      if hide_source:
        cell['metadata']['jupyter']['source_hidden'] = True
        if isinstance(hide_source, str):
          cell['metadata']['jupyter']['source_hidden_str'] = hide_source
      if hide_output:
        cell['metadata']['jupyter']['outputs_hidden'] = True
        if isinstance(hide_output, str):
          cell['metadata']['jupyter']['outputs_hidden_str'] = hide_output

    cell['id'] = str(uuid.uuid4())
    self.nb['cells'].append(cell)
    return cell
  def add_md(self, md='', *, name=None):
    cell = nbf.v4.new_markdown_cell(md)
    if name is not None:
      cell['_name'] = name
    cell['id'] = str(uuid.uuid4())
    self.nb['cells'].append(cell)
    return cell
  def modify_code(self, name, code, *, scrolled=None, collapsed=None, upsert=True, hide_source=None, hide_output=None):
    for cell in self.nb["cells"]:
      if cell.get("_name") == name:
        break
    else:
      if not upsert:
        raise Exception('cell with name {} not exists!'.format(name))
      else:
        cell = self.add_code(code, name=name, scrolled=scrolled, collapsed=collapsed)
    if collapsed is not None:
      cell['metadata']['collapsed'] = collapsed
    if scrolled is not None:
      cell['metadata']['scrolled'] = scrolled
    if hide_source or hide_output:
      cell['metadata'] = {'jupyter':{}}
      if hide_source:
        cell['metadata']['jupyter']['source_hidden'] = True
        if isinstance(hide_source, str):
          cell['metadata']['jupyter']['source_hidden_str'] = hide_source
      if hide_output:
        cell['metadata']['jupyter']['outputs_hidden'] = True
        if isinstance(hide_output, str):
          cell['metadata']['jupyter']['outputs_hidden_str'] = hide_output
    cell['source'] = code
    return cell
  def modify_md(self, name, md, *, upsert=True):
    for cell in self.nb["cells"]:
      if cell.get("_name") == name:
        break
    else:
      if not upsert:
        raise Exception('cell with name {} not exists!'.format(name))
      else:
        cell = self.add_md(md, name=name)
        return cell
    cell['source'] = md
    return cell
  def save(self, file=None):
    if file is None:
      if self.file is None:
        raise Exception('new notebook, should give a file path to save!')
      file = self.file
    with open(file, 'w') as f:
      f.write(json.dumps(self.nb))
  def export(self, file=None, execute=False):
    if file is None:
      if self.file is None:
        raise Exception('should give a filename to export')
      else:

        if self.file.endswith('.ipynb'):
          file = self.file[:-5]+'html'
        elif self.file.endswith('.ipynb-custom'):
          file = self.file[:-12]+'html'
        else:
          raise Exception('nb file is {}, not a ipynb or ipynb-custom file, can not determine the export file'.format(self.file))
    self.econfig = c = Config()

    if execute:
      prep = nbconvert.preprocessors.ExecutePreprocessor()
      c.HTMLExporter.preprocessors.append(prep)

    cus = CustomPreprocessor()
    c.HTMLExporter.preprocessors.append(cus)

    c.HTMLExporter.template_name = 'fmajor'
    self.ee = html_exporter = MyHTMLExporter(config=c)
    #html_exporter.template_name = 'lab'
    self.body, self.resources = html_exporter.from_notebook_node(self.nb)

    with open(file, 'w') as f:
      f.write(self.body)

if __name__ == '__main__':
  if 'test step running' and 1:
    run_good = 0
    class S(StepRunning):
      class first_step:
        def skip(self):
          return True
        def run(self):
          self.l.debug('debug from first step')
          print('from first step 0')
          print('from first step 1')
          print('from first step 23\n45')
      class second_step:
        def skip(self):
          return False
        def run(self):
          print('from second step')
          pass
      class third_step:
        def run(self):
          print('from third step')
          if not run_good:
            raise
      class last_step:
        def run(self):
          print('from last step')
          if not run_good:
            raise
    s = S()
    s._run()
  if 'test nb' and 0:
    nb = NB()
    nb.add_code(textwrap.dedent("""
    import plotly
    import plotly.offline as py
    import plotly.io as pio
    pio.renderers.default='notebook'
    py.init_notebook_mode(connected=False)
    import plotly.graph_objs as go
    from IPython.display import HTML
    fig = go.Figure(go.Scatter(x=[0,1,2,0], y=[0,2,0,0], fill="toself"))
    fig
    """), hide_source=True, hide_output=True)
    nb.export('./nb.html', execute=True)
    #ipynb = '/Users/wujinnnnn/jobs/PKU/CSST/VideoCourse/grizli-tutorials/workspace/demo-cluster/7_spectrum.ipynb'
    eenb = NB.load(ipynb)
    #enb.export('./enb.html')
