from setuptools import setup, Extension

#try:
#	import numpy
#except ImportError:
#	raise Exception("numpy is requiered")

try:
	import numpy
	
	npy_include_dir = numpy.get_include()

	incl_dirs = [npy_include_dir,]
	libs = ['m']


	calcGrid = Extension(   'gadget.calcGrid',
            include_dirs = incl_dirs,
            libraries    = libs,
            sources = ['libs/calcGrid.c','libs/sph.c', 'libs/dg.c'])

	ext_modules = [calcGrid,]
except Exception as inst:
	print(inst)
	print("can not build c module")
	ext_modules = []

setup(name='gadget',
      version='0.1',
      packages=['gadget', 'arepo'],
      ext_modules = ext_modules,
      scripts = ['scripts/arepoinfo', 'scripts/arepoconvert']
      )

