from distutils.core import setup,Extension

try:
	import numpy
except ImportError:
	raise Exception("numpy is requiered")

npy_include_dir = numpy.get_include()

incl_dirs = [npy_include_dir,]
libs = ['m']


calcGrid = Extension(   'gadget.calcGrid',
            include_dirs = incl_dirs,
            libraries    = libs,
            sources = ['libs/calcGrid.c','libs/sph.c'])


setup(name='gadget',
      version='0.1',
      packages=['gadget'],
      ext_modules = [calcGrid,],
      )
