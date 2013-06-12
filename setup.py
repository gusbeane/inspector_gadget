from distutils.core import setup,Extension

incl_dirs = ['/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy',
             '/usr/local/include']
libs_dirs = ['/usr/local/lib']
libs = ['m']


calcGrid = Extension(   'gadget.calcGrid',
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
                    sources = ['libs/calcGrid.c','libs/sph.c'])


setup(name='gadget',
      version='0.1',
      packages=['gadget'],
      ext_modules = [calcGrid,],
      )
