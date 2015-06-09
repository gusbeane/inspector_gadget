import numpy as np
import os.path as path
import struct
import time
import os

import re

import gadget
import gadget.fields as fields

def handlesfile(filename, snapshot=None, filenum=None, snapprefix = "snap", snap = None, **param):
    fname, filenum, snapshot = getFilename(filename, snapshot, filenum, snapprefix, snap)
    if fname is not None:
        f = None
        try:
            swap, endian = endianness_check(fname)
            
            f = open(fname , 'r' )
            f.seek( 0, 0 )
            fheader, = struct.unpack( endian + "i", f.read(4) )
            f.seek( fheader, 1 )
            ffooter, = struct.unpack( endian + "i", f.read(4) )
            f.close()
            
            if fheader == ffooter:
                return True
            else:
                return False
        except:
            if f is not None:
                f.close()
            return False
    else:
        return False
    
def writefile(filename):
    return False
    
def getFilename(filename, snapshot=None, filenum=None, snapprefix = "snap", snap=None):
    if snapshot is not None:
        filename = re.sub(r"snapdir_[0-9]+",r"snapdir_%3d"%snapshot, filename)
        filename = re.sub(r"%s_[0-9]+\."%snapprefix,r"%s_%3d."%(snapprefix,snapshot), filename)
    
    if filenum is not None:
        filename = re.sub(r"\.[0-9]+$",r".%d"%filenum, filename)
        
    if filenum is None:
        filenum = 0
        
    if path.isfile( filename ):
        filename = filename  
    elif  path.isfile( filename + ".0" ):
        filename += ".0"
    elif snapshot is not None and path.isfile( filename + snapprefix +"_%3d"%snapshot):
        filename = filename + snapprefix +"_%3d"%snapshot
    elif snapshot is not None and path.isfile( filename + "snapdir_%3d/"%snapshot + snapprefix +"_%3d"%snapshot + ".%d"%filenum):
        filename = filename + "snapdir_%3d/"%snapshot + snapprefix +"_%3d"%snapshot + ".%d"%filenum
    else:
        return (None, None, None)
        
    res = re.findall(r"_([0-9]+)(\.([0-9]*))?$", filename)
    
    if len(res) > 0:
        if res[0][2] != "":
            filenum = int(res[0][2])
        else:
            filenum = None
    
        if res[0][0] != "":
            snapshot = int(res[0][0])
        else:
            snapshot = None
    else:
        filenum = None
        snapshot = None
           
    return (filename, filenum, snapshot)

class Format1:
    def __init__( self, sn, snapprefix=None, block_sequence=['pos', 'vel', 'id', 'mass', 'u']):
        self.sn = sn

        if not isinstance(self.sn, gadget.loader.Snapshot):
            raise Exception("Format 1 can only load snapshots")
        
        self.block_sequence = [self.sn._normalizeName(i.strip().lower()) for i in block_sequence]
        
        self.snapprefix = snapprefix
        
    def load(self, num, snapshot):
        self.sn.npart_loaded = np.zeros(6,dtype=np.longlong)
        
        if num is None:
            num = self.sn.filenum
            
        if snapshot is None:
            snapshot = self.sn.snapshot
        
        filename, num, snapshot = getFilename(self.sn.filename, snapshot, num, self.snapprefix, self.sn)
        
        self.files = [filename]
        self.filecount = 1
        if self.sn._combineFiles:
            while path.exists( filename + ".%d" % self.filecount ):
                self.files += [filename + ".%d" % self.filecount]
                self.filecount += 1
        
        self.load_header( 0, verbose=self.sn._verbose )

        if not self.sn._onlyHeader:
            self.load_data()
            
        if self.sn._combineFiles:
            self.sn.filenum = None
        else:
            self.sn.filenum = num
            
        self.sn.snapshot = snapshot
        self.sn.filename = filename
            
        self.sn._path = path.abspath(self.sn.filename)

    def load_header( self, fileid, verbose=False ):
        swap, endian = endianness_check( self.files[fileid] )
        
        f = open( self.files[fileid], 'r' )

        f.seek( 4, 1 ) # skip fortran header of data block
        s = f.read(24)
        self.sn.NumPart_ThisFile = np.longlong( struct.unpack( endian + "6i", s ) )
        s = f.read(48)
        self.sn.MassTable = np.array( struct.unpack( endian + "6d", s ) )
        s = f.read(24)
        self.sn.Time, self.sn.Redshift, self.sn.Flag_Sfr, self.sn.Flag_Feedback = struct.unpack( endian + "ddii", s )
        s = f.read(24)
        self.sn.NumPart_Total = np.longlong( struct.unpack( endian + "6i", s ) )
        s = f.read(16)
        self.sn.Flag_Cooling, self.sn.NumFilesPerSnapshot, self.sn.BoxSize = struct.unpack( endian + "iid", s )
        s = f.read(24)
        self.sn.Omega0, self.sn.OmegaLambda, self.sn.HubbleParam = struct.unpack( endian + "ddd", s )
        s = f.read(8)
        self.sn.Flag_StellarAge, self.sn.Flag_Metals = struct.unpack( endian + "ii", s )
        s = f.read(24)
        self.sn.NumPart_Total_HighWord = np.longlong( struct.unpack( endian + "6i", s ) )<<32
        s = f.read(12)
        self.sn.Flag_EntropyInsteadU, self.sn.Flag_DoublePrecision, self.sn.Flag_lpt_ics = struct.unpack( "iii", s )
        s = f.read(52)
        
        self.sn._headerfields = ['Time', 'Redshift', 'BoxSize', 'Omega0', 'OmegaLambda', 'HubbleParam', 'Flag_Sfr',
                          'Flag_Feedback', 'Flag_Cooling', 'Flag_StellarAge', 'Flag_Metals', 'Flag_DoublePrecision', 
                          'NumPart_ThisFile', 'NumPart_Total', 'NumPart_Total_HighWord', 'NumFilesPerSnapshot', 'MassTable', 'Flag_lpt_ics', 'Flag_EntropyInsteadU' ]

        self.sn.nparticlesall = np.longlong(self.sn.NumPart_Total)
        self.sn.nparticlesall += np.longlong(self.sn.NumPart_Total_HighWord)<<32

        if self.sn.nparticlesall.sum() == 0:
            self.sn.nparticlesall = self.sn.NumPart_ThisFile
                
        sort = self.sn.nparticlesall.argsort()
        if self.sn.nparticlesall[ sort[::-1] ][1] == 0:
            self.sn.singleparticlespecies = True
        else:
            self.sn.singleparticlespecies = False


        f.close()

        self.sn.npart = self.sn.NumPart_ThisFile.sum()
        self.sn.npartall = self.sn.nparticlesall.sum()

        if verbose:
            print("nparticlesall:", self.sn.nparticlesall, "sum:", self.sn.npartall)

        if fileid == 0:
            if (self.sn.NumFilesPerSnapshot != self.filecount) and not (self.sn.NumFilesPerSnapshot == 0 and self.filecount == 1):
                raise Exception( "Number of files detected (%d) and NumFilesPerSnapshot in the header (%d) are inconsistent." % (self.filecount, self.sn.NumFilesPerSnapshot) )
        if verbose:
            print("Snapshot contains %d particles." % self.sn.npartall)
        return

    def load_data( self ):
        swap, endian = endianness_check( self.files[0] )
        self.sn.data = {}

        nparttot = np.zeros( 6, dtype='int32' )    
        for fileid in range( self.filecount ):
            self.load_header( fileid, verbose=self.sn._verbose )
            
            f = open( self.files[fileid], 'r' )
            
            #skip header
            fheader, = struct.unpack( endian + "i", f.read(4) )
            f.seek( fheader, 1 )
            ffooter, = struct.unpack( endian + "i", f.read(4) )
                    
            for block in self.block_sequence:
                npart, npartall = self.get_block_size_from_table(block)

                if npart == 0:
                    continue
                                                           
                s = f.read(4)
                if len(s) == 0:
                    break
                fheader, = struct.unpack( endian + "i", s ) 
                               
                if self.sn._fields is not None and not block in self.sn._fields:
                    f.seek( fheader, 1 )
                    ffooter, = struct.unpack( endian + "i", f.read(4) )
                    if fheader != ffooter:
                        raise Exception("Bad field: fheader %d, ffooter %d\n"%(fheader.ffooter))
                    
                    if self.sn._verbose:
                        print("Skipping block %s"%(block))
                    continue

                if self.sn._verbose:
                    print("Loading block %s of file %s." % (block, fileid))

                if block in ['id',"nonn"]:
                    blocktype = "i4"
                    elementsize = 4
                else:
                    if self.sn.Flag_DoublePrecision:
                        blocktype = 'f8'
                        elementsize = 8
                    else:
                        blocktype = 'f4'
                        elementsize = 4
                
                if blocktype == 'f4' and self.sn._toDouble:
                    datatype = "float64"
                else:
                    datatype = blocktype
                 
                nsum = 0
                for i in np.arange(6):
                    if not self.parttype_has_block(i,block):
                        continue
                    
                    npart, npartall = self.get_block_size_from_table(block)
                    dim = self.get_block_dim_from_table(block)
                    npartptype = self.sn.NumPart_ThisFile[i]
                    elements = npartptype * dim
                    pres = self.sn._learnPresent(block,gr=i,shape=dim)
                        
                    if self.sn._verbose:
                        print("Loading block %s, type %d, elements %d, dimension %d, particles %d/%d."%(block, i, elements, dim, npartptype, npartall))
                    
                    if not block in self.sn.data:
                        if dim == 1:
                            self.sn.data[block] = np.zeros(npartall, dtype=datatype)
                        else:
                            self.sn.data[block] = np.zeros((npartall, dim), dtype=datatype)
    
    
                    lb = nsum + nparttot[i]
                    ub = lb + npartptype
                    
                    if self.sn._verbose:
                        print("Block contains %d elements (elementsize=%g, lb=%d, ub=%d)." % (elements,elementsize,lb,ub))
    
                    if dim == 1:
                        self.sn.data[block][lb:ub] = np.fromfile(f, dtype=endian+blocktype, count=elements)
                    else:
                        self.sn.data[block][lb:ub,:] = np.fromfile(f, dtype=endian+blocktype, count=elements).reshape( npartptype, dim )
                    
                    nsum += self.sn.nparticlesall[i]
                    
                ffooter, = struct.unpack( endian + "i", f.read(4) )
                if fheader != ffooter:
                    raise Exception("Bad field: fheader %d, ffooter %d\n"%(fheader,ffooter))
  
            nparttot += self.sn.NumPart_ThisFile
            s = f.read(1)
            if len(s) != 0:
                print("Warning end of file '%s' not reached yet, but unknowen fields ahead. Thus some items might be missing on the snapshot object."%self.files[fileid])
            f.close()
        
        if self.filecount > 1:
            self.sn.npart = self.sn.npartall

        self.sn.npart_loaded = self.sn.nparticlesall
        
        if self.sn._verbose:
            print('%d particles loaded.' % self.sn.npartall)

    def close(self):
        pass
    
    def write(self):
        raise Exception("Writing of format 1 files is not supported")
    
    def get_block_size_from_table( self, block ):
        npart = 0
        npartall = 0
        if block in ['pos', 'vel', 'id', 'pot', 'tstp']:
            # present for all particles
            npart = self.sn.NumPart_ThisFile.sum()
            npartall = self.sn.nparticlesall.sum()
        elif block in ['u', 'rho', 'ne', 'nh', 'hsml', 'sfr', 'z', 'xnuc', 'pres', 'vort', 'vol', 'hrgm', 'ref', 'divv', 'rotv', 'dudt', 'bfld', 'divb', 'psi', 'ref', 'pass', 'temp', 'grar', 'grap', 'nonn', 'csnd']:
            # present for hydro particles
            npart = self.sn.NumPart_ThisFile[0]
            npartall = self.sn.nparticlesall[0]
        elif block in ["age", "z"]:
            # present for star particles
            npart = self.sn.NumPart_ThisFile[4]
            npartall = self.sn.nparticlesall[4]
        elif block in ["bhma", "bhmd"]:
            #present for black hole particles
            npart = self.sn.NumPart_ThisFile[5]
            npartall = self.sn.nparticlesall[5]
        elif block in ["mass"]:
            npart = self.sn.NumPart_ThisFile[np.where(self.sn.MassTable==0)[0]].sum()
            npartall = self.sn.nparticlesall[np.where(self.sn.MassTable==0)[0]].sum()
        else:
            raise Exception("Unknown block: %s. Add to format 1 loader")
        
        return npart, npartall
    
    def parttype_has_block( self, ptype, block ):
        if block in ['pos', 'vel', 'id', 'pot', 'tstp']:
            return True
        elif block in ['mass']:
            if self.sn.MassTable[ptype] == 0:
                return True
            else:
                return False
        elif block in ['u', 'rho', 'ne', 'nh', 'hsml', 'sfr', 'z', 'xnuc', 'pres', 'vort', 'vol', 'hrgm', 'ref', 'divv', 'rotv', 'dudt', 'bfld', 'divb', 'psi', 'mass', 'pass', 'temp', 'acce', 'p', 'grar', 'grap', 'nonn', 'csnd']:
            if ptype == 0:
                return True
            else:
                return False
        elif block in ["age", "z"]:
            if ptype == 4:
                return True
            else:
                return False
        elif block in ["bhma", "bhmd"]:
            if ptype == 5:
                return True
            else:
                return False
        else:
            raise Exception("Unknown block: %s. Add to format 1 loader")
        
    def get_block_dim_from_table( self, block ):
        if block in ['pos', 'vel', 'divv', 'vort', 'bfld', 'grar', 'grap']:
            return 3
        else:
            return 1

def endianness_local():
    s = struct.pack( "bb", 1, 0 )
    r, = struct.unpack( "h", s )

    if r == 1:
        return '<'
    else:
        return '>'

def endianness_check(filename ):
    if not path.exists( filename ):
        print("File %s does not exist." % filename)
        return False

    filesize = path.getsize( filename )

    endian_local = endianness_local()
    endian_data = endian_local

    f = open( filename )
    s = f.read(4)
    if len(s) > 0:
        size, = struct.unpack( "<i", s )
        size = abs( size )

        if size < filesize:
            f.seek( size, 1 )
            s = f.read(4)
        else:
            s = ""
        
        if (len(s) > 0) and (struct.unpack( "<i", s )[0] == size):
            f.close()
            endian_data = "<"
        else:
            f.seek( 0, 0 )
            size, = struct.unpack( ">i", f.read(4) )
            size = abs( size )
            
            if size < filesize:
                f.seek( size, 1 )
                s = f.read(4)
            else:
                s = ""
            
            if (len(s) > 0) and (struct.unpack( ">i", s )[0] == size):
                f.close()
                endian_data = ">"
            else:
                f.close()
                raise Exception("Format 1: File %s is corrupt." % filename)
                return False
    else:
        f.close()
        if self.sn._verbose:
            print("File %s is empty." % filename)

    return (endian_local != endian_data, endian_data)