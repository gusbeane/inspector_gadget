# -*- coding: utf-8 -*-

import numpy as np
import numbers
import copy


class Quantity(np.ndarray):

    def __new__(cls, input_array, unit=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.unit = unit
        
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comment
        if obj is None: return
        self.unit = getattr(obj, 'unit', None)
        
    def __array_wrap__(self, out_arr, context=None):
        #binary operations
        if context is not None and len(context[1]) >= 2 :
            a = context[1][0]
            b = context[1][1]
      
            if context[0] == np.multiply:           
                if type(b) == Quantity:
                    t = a
                    a = b
                    b = t
                if type(b) == Quantity:
                    out_arr.unit = a.unit * b.unit
                    
            elif context[0] == np.divide:
                if type(a) == Quantity and type(b) == Quantity:
                    out_arr.unit = a.unit / b.unit
                    
            elif context[0] == np.add or context[0] == np.subtract:
                if type(a) == Quantity and type(b) == Quantity:
                    if a.unit._check(b.unit):
                        pass
                    else:
                        print "Warning unit missmatch: [" + str(a.unit) + "] vs. [" + str(b.unit) + "]"
                        out_arr.unit = None
                
            elif context[0] == np.power:
                out_arr.unit = context[1][0].unit**context[1][1]
                    
            else:
                out_arr.unit = None
                
        #unary operations
        elif context is not None and len(context[1]) == 1:
            if context[0] == np.sqrt:
                out_arr.unit = context[1][0].unit**0.5
            elif context[0] == np.square:
                out_arr.unit = context[1][0].unit**2.
            elif context[0] == np.absolute:
                out_arr.unit = context[1][0].unit
            elif context[0] == np.negative:
                out_arr.unit = context[1][0].unit
            else:
                out_arr.unit = None
        else:
                out_arr.unit = None
                
        return np.ndarray.__array_wrap__(self, out_arr, context)
    
    def __getitem__(self, key):
        try:
            out = super(Quantity, self).__getitem__(key)
        except IndexError:
            if self.isscalar:
                raise TypeError(
                    "'{cls}' object with a scalar value does not support "
                    "indexing".format(cls=self.__class__.__name__))
            else:
                raise

        if type(out) is not type(self):
            if not isinstance(out, np.ndarray):
                out = np.array(out)

            out = out.view(self.__class__)
            out.__array_finalize__(self)
            out.unit = copy.copy(self.unit)
            
        return out
    
    @property
    def isscalar(self):
        return not self.shape

    
    def __repr__(self):
        if self.shape == ():
            return np.array2string(self.view(np.ndarray)) + " " + self.unit.__repr__()
        else:
            return np.ndarray.__repr__(self) + " unit: " + self.unit.__repr__()
        
    @property
    def value(self):
        value = self.view(np.ndarray)
        if self.shape:
            return value
        else:
            return value.item()
        
    def to_physical(self, a=None, h=None):
        factor = 1.
        
        if not self.unit._a_applied and a is not None:
            factor *= a**self.unit.a
            self.unit._a_applied = True
            
        if not self.unit._h_applied and h is not None:
            factor *= h**self.unit.h
            self.unit._h_applied = True
            
        if factor != 1.:        
            self *= factor
            
    def as_physical(self, a=None, h=None):
        factor = 1.
        
        unit = copy.copy(self.unit)
        
        if not unit._a_applied and a is not None:
            factor *= a**unit.a
            unit._a_applied = True
            
        if not unit._h_applied and h is not None:
            factor *= h**unit.h
            unit._h_applied = True
                  
        out = self * factor
        out.unit = unit
        
        return out
            
    def to_comoving(self, a=None, h=None):
        factor = 1.
        
        if self.unit._a_applied and a is not None:
            factor /= a**self.unit.a
            self.unit._a_applied = False
        elif not self.unit._a_applied and a is not None:
            raise Exception("can not convert to comoving a, because unit is already comoving")
            
        if self.unit._h_applied and h is not None:
            factor /= h**self.unit.h
            self.unit._h_applied = False
        elif not self.unit._h_applied and h is not None:
            raise Exception("can not convert to comoving h, because unit is already comoving")
            
        if factor != 1.:        
            self *= factor

    def as_comoving(self, a=None, h=None):
        factor = 1.
        
        unit = copy.copy(self.unit)
        
        if unit._a_applied and a is not None:
            factor /= a**unit.a
            unit._a_applied = False
        elif not unit._a_applied and a is not None:
            raise Exception("can not convert to comoving a, because unit is already comoving")
            
        if nit._h_applied and h is not None:
            factor /= h**unit.h
            unit._h_applied = False
        elif not unit._h_applied and h is not None:
            raise Exception("can not convert to comoving h, because unit is already comoving")
                  
        out = self * factor
        out.unit = unit
        
        return out
    
    def to_unit(self, unit):
        if not _equiv(self.unit.L, unit.L):
            print "unit error in length scaling"
            return
        
        if not _equiv(self.unit.M, unit.M):
            print "unit error in mass scaling"
            return
        
        if not _equiv(self.unit.V, unit.V):
            print "unit error in velocity scaling"
            return
        
        factor = self.unit.c/unit.c
        
        self *= factor
        
        self.unit.c = unit.c
        
    def as_unit(self, unit):
        if not _equiv(self.unit.L, unit.L):
            print "unit error in length scaling"
            return
        
        if not _equiv(self.unit.M, unit.M):
            print "unit error in mass scaling"
            return
        
        if not _equiv(self.unit.V, unit.V):
            print "unit error in velocity scaling"
            return
        
        factor = self.unit.c/unit.c
        
        out = self * factor
        
        out.unit = copy.copy(self.unit)
        out.unit.c = unit.c
        
        return out

        
    def to_unit_system(self, system):
        factor = system.lunit**self.unit.L * system.munit**self.unit.M * system.vunit**self.unit.V
        
        self *= self.unit.c/factor
        
        self.unit.c = factor
        
    def as_unit_system(self, system):
        factor = system.lunit**self.unit.L * system.munit**self.unit.M * system.vunit**self.unit.V
        
        out = self * self.unit.c/factor
        
        out.unit = copy.copy(self.unit)
        out.unit.c = factor
        
        return out
        
def _equiv(a,b):
    if a == 0 and b == 0:
        return True
    if ( a==0. or b==0. ) and a!=b:
        return False
    
    if( np.abs(a-b) / b) < 1e-6:
        if( np.abs(a-b) / a) < 1e-6:
            return True
        
    return False

def _print_exp(e):
    if e.is_integer():
        return str(int(e))
    else:
        return str(e)
    

class UnitSystem(object):
    def __init__(self, lunit, munit, vunit, name=None):
        self.lunit = lunit
        self.munit = munit
        self.vunit = vunit
        
        self.name = name

class Unit(object):
    
    def __init__(self, a, h, L, M, V, c, name = None, short_name = None):
        
        self.a = float(a)
        self.h = float(h)
        self.L = float(L)
        self.M = float(M)
        self.V = float(V)
        self.c = float(c)
        
        self.name = name
        self.short_name = short_name
        
        if self.h != 0:
            self._h_applied = False
        else:
            self._h_applied = True
            
        if self.a != 0:
            self._a_applied = False
        else:
            self._a_applied = True
             
        
    def __repr__(self, latex = False):
        result = None
        
        shortest = None
        dist = np.inf
        
        for u in Units:
            if u.L == self.L and u.M == self.M and u.V == self.V:
                if _equiv(self.c, u.c):
                    result = u.short_name
                    break
                elif np.abs(np.log10(self.c/u.c)) < dist:
                    dist = np.abs(np.log10(self.c/u.c))
                    shortest = u
                      
        if result is None and shortest is not None:
            result = "%g"%(self.c/shortest.c) + " " + shortest.short_name
        
        if result is None:
            distlog = np.inf
            distpower = np.inf
            
            l = self.L + self.V
            m = self.M
            t = -self.V
            
            if _equiv(l, 0.):
                lunits = [LengthUnits[0]]
            else:
                lunits = LengthUnits
                
            if _equiv(m, 0.):
                munits = [MassUnits[0]]
            else:
                munits = MassUnits
               
            if _equiv(t, 0.):
                tunits = [LengthUnits[0]]
            else:
                tunits = TimeUnits
            
            for lu in lunits:
                for mu in munits:
                    for tu in tunits:
                        factor = lu.c**l * mu.c**l * tu.c**t
                        
                        f = self.c/factor
                        d = np.abs(np.log10(f))
                        dl = d - np.floor(d)
                        dp = np.floor(d)
                        if _equiv(dl,distlog):
                            if dp < distpower:
                                distlog = dl
                                distpower = dp
                                shortest = (lu,mu,tu, f)
                        elif dl < distlog:
                            distlog = dl
                            distpower = dp
                            shortest = (lu,mu,tu, f)
                         
            if shortest is not None:
                if not _equiv(1., shortest[3]):
                    result = "%g"%(shortest[3]) + " "
                else:
                    result = "" 
                    
                f = [l, m, t]
                for i in np.arange(3):
                    if f[i] != 0.:
                        if _equiv(1.,f[i]):
                            result += shortest[i].short_name + " "
                        else:
                            result += shortest[i].short_name + "^" + _print_exp(f[i]) + " "
                            
                if len(result) > 0 and result[-1] == " ":
                    result = result[:-1]
            
        if result is None:
            result = "%g"%(self.c) + " [cm]^" + _print_exp(self.L) + " [g]^" + _print_exp(self.M) + " [cm/s]^" + _print_exp(self.V)
            
        if not self._a_applied and self.a != 0:
            result = "a^" + _print_exp(self.a) + " " + result
        
        if not self._h_applied and self.h != 0:
            result = "h^" + _print_exp(self.h) + " " + result
                
        return result
    
    def __mul__(self, other):
        if type(other) == Unit:
            return Unit(self.a+other.a, self.h+other.h, self.L+other.L, self.M+other.M, self.V+other.V, self.c*other.c)
        elif isinstance(other,numbers.Number):
            return Unit(self.a, self.h, self.L, self.M, self.V, self.c*other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    
    def __div__(self, other):
        if type(other) == Unit:
            return Unit(self.a-other.a, self.h-other.h, self.L-other.L, self.M-other.M, self.V-other.V, self.c/other.c)
        elif isinstance(other,numbers.Number):
            return Unit(self.a, self.h, self.L, self.M, self.V, self.c/other)
               
    def __rdiv__(self, other):
        if isinstance(other,numbers.Number):
            return Unit(-self.a, -self.h, -self.L, -self.M, -self.V, other/self.c)
        
    def __pow__(self, other):
        if isinstance(other,numbers.Number):
            return Unit(self.a*other, self.h*other, self.L*other, self.M*other, self.V*other, self.c**other)
        
    def _check(self, other, comoving=True):
        if _equiv(self.L, other.L)  and _equiv(self.M, other.M)  and _equiv(self.V, other.V)  and _equiv(self.c, other.c):
            if comoving:
                if _equiv(self.a, other.a) and _equiv(self.h, other.h):
                    return True
                else:
                    return False
            else:
                return True
        
        return False
        

#length units
cm = centimeter = Unit(0., 0., 1., 0., 0., 1., "centimeter", "cm")
m = meter = Unit(0., 0., 1., 0., 0., 1e2, "meter", "m")
km = kilometer = Unit(0., 0., 1., 0., 0., 1e5, "kilometer", "km")
pc = parsec = Unit(0., 0., 1., 0., 0., 3.085678e18, "parsec", "pc")
kpc = kiloparsec = Unit(0., 0., 1., 0., 0., 3.085678e21, "kiloparsec", "kpc")
mpc = megaparsec = Unit(0., 0., 1., 0., 0., 3.085678e24, "megaparsec", "mpc")
A = angstrom = Unit(0., 0., 1., 0., 0., 1e-8, "angstro,", u"Å ")
nm = nanometer = Unit(0., 0., 1., 0., 0., 1e-9, "nanometer,", "nm")
um = micrometer = Unit(0., 0., 1., 0., 0., 1e-6, "micrometer,", u"μm")
AU = astronomicalunit = Unit(0., 0., 1., 0., 0., 14959787070000., "astronomical unit,", "AU")


LengthUnits = [centimeter, meter, kilometer, parsec, kiloparsec, megaparsec, angstrom, nanometer, micrometer, AU]

g = gram = Unit(0., 0., 0., 1., 0., 1., "gram", "g")
kg = kilogram = Unit(0., 0., 0., 1., 0., 1e3, "kilogram", "kg")
t = tonne = Unit(0., 0., 0., 1., 0., 1e6, "tonne", "t")
msol = msolar = Unit(0., 0., 0., 1., 0., 1.989e33, "solarmass", "Msol")

MassUnits = [gram, kilogram, tonne, msolar]

kms = Unit(0., 0., 0., 0., 1., 1e5, "kilometer per second", "km/s")
cms = Unit(0., 0., 0., 0., 1., 1., "centimeter per second", "cm/s")
ms = Unit(0., 0., 0., 0., 1., 100., "meter per second", "m/s")
kmh = Unit(0., 0., 0., 0., 1., 1e5/3600., "kilometer per hour", "km/h")

VelocityUnits = [kms, ms, cms,kmh]

s = second = Unit(0., 0., 1.,0.,-1., 1., "second","s")
min = minute = Unit(0., 0., 1.,0.,-1., 60., "minute","min")
hour = Unit(0., 0., 1.,0.,-1., 3600, "hour","a")
d = day = Unit(0., 0., 1.,0.,-1., 24*3600., "day","d")


TimeUnits = [second, minute, hour, day]

erg = Unit(0., 0., 0., 1., 2., 1., "erg", "erg")
joule = Unit(0., 0., 0., 1., 2., 1e7, "joule", "J")

EnergyUnits = [erg, joule]


Units = []
Units.extend(LengthUnits)
Units.extend(MassUnits)
Units.extend(VelocityUnits)
Units.extend(EnergyUnits)
Units.extend(TimeUnits)
cgs = UnitSystem(1.,1.,1., "cgs")
mks = UnitSystem(100.,1000.,100., "mks")


h = Unit(0.,1.,0.,0.,0.,1.,'h','h')
a = Unit(1.,0.,0.,0.,0.,1.,'a','a')

