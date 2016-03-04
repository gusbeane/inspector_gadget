# -*- coding: utf-8 -*-

import numpy as np
import numbers


import gadget.units as u

try:
    
    c = speedoflight = u.Quantity(299792458., unit=u.meter/u.second)
    pc = parsec = u.Quantity(1., unit=u.parsec)
    kpc = kiloparsec = u.Quantity(1., unit=u.kiloparsec)
    mpc = megaparsec = u.Quantity(1., unit=u.megaparsec)
    AU = astronomicalunit = u.Quantity(1., unit=u.astronomicalunit)
    H = Hubble = u.Quantity(100., unit=u.kms/u.mpc*u.h)
    
    
    
    
except:
    print("Could not setup constants")
