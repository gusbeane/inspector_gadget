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
            
            f = open(fname, 'r' )
            s = f.read(16)
            while len(s) > 0:
                fheader, name, length, ffooter = struct.unpack( endian + "i4sii", s )
            
                if name != "HEAD":
                    f.seek( length, 1 )
                    s = f.read( 16 )
                else:
                    f.close()
                    return True
            f.close()
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

class Format2:

    def __init__( self, sn, nommap=False, tracer=False, snapprefix=None):       
        self.sn = sn
        
        if not isinstance(self.sn, gadget.loader.Snapshot):
            raise Exception("Format 2 can only load snapshots")
        
        if sn.__fields__ != None:
            self.loadlist = sn.__fields__
        else:
            self.loadlist = []
            
        self.nommap=nommap
        self.tracer = tracer

        self.datablocks_skip = ["HEAD"]
        self.datablocks_int32 = ["ID  ","NONN"]
        
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
        if self.sn.__combineFiles__:
            while path.exists( filename + ".%d" % self.filecount ):
                self.files += [filename + ".%d" % self.filecount]
                self.filecount += 1

            if not self.nommap and self.filecount > 1:
                print("Multiple files detected, thus mmap is deactivated.")
                self.nommap = True
        
        self.load_header( 0, verbose=self.sn.__verbose__ )
        self.get_blocks( 0, verbose=self.sn.__verbose__ )

        if not self.sn.__onlyHeader__:
            if self.filecount == 1 and not self.nommap and not self.sn.__toDouble__:
                self.load_data_mmap()
            else:
                self.load_data()
            
        if self.sn.__combineFiles__:
            self.sn.filenum = None
        else:
            self.sn.filenum = num
            
        self.sn.snapshot = snapshot
        self.sn.filename = filename
            
        self.sn.__path__ = path.abspath(self.sn.filename)

    def get_blocks( self, fileid, verbose=False ):
        swap, endian = endianness_check( self.files[fileid] )
		
        self.blocks = {}
        self.origdata = []
        f = open( self.files[fileid], 'r' )
		
        fpos = f.tell()
        s = f.read(16)
        while len(s) > 0:
            fheader, name, length, ffooter = struct.unpack( endian + "i4sii", s )
            self.blocks[ name ] = fpos
            

            for i in np.arange(0,6):
                if self.parttype_has_block(i,name):
                    self.sn.__learnPresent__(name.strip().lower(), gr=i, shape=self.get_block_dim_from_table(name))

            length = self.get_block_length( fileid, fpos + 16, name, length, endian, verbose=verbose )
            if length < 0:
                return False

            if verbose:
                print("Block %s, length %d, offset %d." %(name, length, fpos))

            f.seek( length, 1 )
            fpos = f.tell()
            s = f.read(16)
			
            self.origdata += [name]

        if verbose:
            print("%d blocks detected." % len(self.blocks.keys()))
        return True

    def get_block_length( self, fileid, start, name, length, endian, verbose=False ):
        if verbose:
            print("Getting block length of block %s." % (name))
            
        if self.check_block_length( fileid, start, length, endian ):
            return length

        if verbose:
            print("First check failed.")

        if name == "XNUC" or name == "PASS":
            bs, bsall = self.get_block_size_from_table( name )
            if self.sn.Flag_DoublePrecision:
                ele = 8
            else:
                ele = 4

            for dim in range( 500 ):
                length = bs * dim * ele + 8
                if self.check_block_length( fileid, start, length, endian ):
                    if verbose:
                        print("Block %s solved, bs=%d, dim=%d." % (name, bs, dim))
                    return length

            print("Error determining the length of block %s." % (name))
            return -1
			
        else:
            bs, bsall = self.get_block_size_from_table( name )
            dim = self.get_block_dim_from_table( name )
            if self.sn.Flag_DoublePrecision:
                ele = 8
            else:
                ele = 4

            length = bs * dim * ele + 8
			
            if self.check_block_length( fileid, start, length, endian ):
                return length
            else:
                print("Error determining the length of block %s." % (name))
                return -1
		
        return length

    def check_block_length( self, fileid, start, length, endian ):
        f = open( self.files[fileid], 'r' )

        f.seek( start, 0 )
        fheader, = struct.unpack( endian + "i", f.read(4) )
        f.seek( length-8, 1 )
        ffooter, = struct.unpack( endian + "i", f.read(4) )
        if fheader != ffooter:
            f.close()
            return False

        s = f.read(4)
        if len( s ) == 4:
            fheader, = struct.unpack( endian + "i", s )
            if fheader != 8:
                f.close()
                return False

        f.close()
        return True

    def load_header( self, fileid, verbose=False ):
        swap, endian = endianness_check( self.files[fileid] )
		
        f = open( self.files[fileid], 'r' )
        s = f.read(16)
        while len(s) > 0:
            fheader, name, length, ffooter = struct.unpack( endian + "i4sii", s )
            
            if name != "HEAD":
                f.seek( length, 1 )
                s = f.read( 16 )
            else:
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
                
                self.sn.__headerfields__ = ['Time', 'Redshift', 'BoxSize', 'Omega0', 'OmegaLambda', 'HubbleParam', 'Flag_Sfr',
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
                s = "" # stop reading

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

    def load_data_mmap( self ):
        # memory mapping works only if there is only one file
        swap, endian = endianness_check( self.files[0] )
        self.sn.data = {}

        self.get_blocks( 0, verbose=self.sn.__verbose__ )
        f = open( self.files[0], 'r' )
        for block in self.blocks.keys():
            # skip some blocks
            if block in self.datablocks_skip:
                continue

            if len(self.loadlist) > 0 and not block.strip().lower() in self.loadlist:
                continue


            f.seek( self.blocks[block], 0 )
            fheader, name, length, ffooter = struct.unpack( endian + "i4sii", f.read(16) )
            length = self.get_block_length( 0, f.tell(), name, length, endian )

            if block in self.datablocks_int32:
                blocktype = "i4"
                elementsize = 4
            else:
                if self.sn.Flag_DoublePrecision:
                    blocktype = 'f8'
                    elementsize = 8
                else:
                    blocktype = 'f4'
                    elementsize = 4

            nsum = 0
            for ptype in range( 6 ):
                if self.sn.NumPart_ThisFile[ptype] == 0:
                    continue
                    
                if not self.parttype_has_block( ptype, name ):
                    continue

                nsum += self.sn.nparticlesall[ptype]
                
            elements = (length-8)/elementsize
            dim = elements / nsum
            
            if self.sn.__verbose__:
                print("Loading block %s, offset %d, length %d, elements %d, dimension %d, particles %d/%d." % (block, self.blocks[block], length, elements, dim, nsum, self.sn.nparticlesall.sum()))

            offset = f.tell() + 4
            blockname = block.strip().lower()
            if dim == 1:
                self.sn.data[ blockname ] = np.memmap( self.files[0], offset=offset, mode='c', dtype=endian+blocktype, shape=(nsum) )
            else:
                self.sn.data[ blockname ] = np.memmap( self.files[0], offset=offset, mode='c', dtype=endian+blocktype, shape=(nsum, dim) )

        self.sn.npart_loaded = self.sn.nparticlesall
        return

    def load_data( self ):
        swap, endian = endianness_check( self.files[0] )
        self.sn.data = {}

        nparttot = np.zeros( 6, dtype='int32' )    
        for fileid in range( self.filecount ):
            self.load_header( fileid, verbose=self.sn.__verbose__ )
            self.get_blocks( fileid, verbose=self.sn.__verbose__ )
            
            f = open( self.files[fileid], 'r' )
            for block in self.blocks.keys():
                # skip some blocks
                if block in self.datablocks_skip:
                    continue

                if len(self.loadlist) > 0 and not block.strip().lower() in self.loadlist:
                    continue

                if self.sn.__verbose__:
                    print("Loading block %s of file %s." % (block, fileid))

                f.seek( self.blocks[block], 0 )
                fheader, name, length, ffooter = struct.unpack( endian + "i4sii", f.read(16) )
                length = self.get_block_length( fileid, f.tell(), name, length, endian )
                f.seek( 4, 1 ) # skip fortran header of data field

                if block in self.datablocks_int32:
                    blocktype = "i4"
                    elementsize = 4
                else:
                    if self.sn.Flag_DoublePrecision:
                        blocktype = 'f8'
                        elementsize = 8
                    else:
                        blocktype = 'f4'
                        elementsize = 4
                
                nsum = 0
                for ptype in range( 6 ):
                    if self.sn.NumPart_ThisFile[ptype] == 0:
                        continue
                    
                    if not self.parttype_has_block( ptype, name ):
                        continue
                    
                    npart, npartall = self.get_block_size_from_table( name )
                    dim = self.get_block_dim_from_table( name )
                    npartptype = self.sn.NumPart_ThisFile[ptype]
                    elements = npartptype * dim
                    
                    if self.sn.__verbose__:
                        print("Loading block %s, offset %d, length %d, elements %d, dimension %d, type %d, particles %d/%d." % (block, self.blocks[block], length, elements, dim, ptype, npartptype, npartall))
                    
                    blockname = block.strip().lower()
                    if not blockname in self.sn.data:
                        if dim == 1:
                            self.sn.data[ blockname ] = np.zeros( npartall, dtype=blocktype )
                        else:
                            self.sn.data[ blockname ] = np.zeros( (npartall, dim), dtype=blocktype )

                    if blocktype == 'f4' and self.sn.__toDouble__:
                        self.sn.data[ blockname ] = self.sn.data[ blockname ].astype( 'float64' ) # change array type to float64

                    lb = nsum + nparttot[ptype]
                    ub = lb + npartptype
                    
                    if self.sn.__verbose__:
                        print("Block contains %d elements (length=%g, elementsize=%g, lb=%d, ub=%d)." % (elements,length,elementsize,lb,ub))

                    if dim == 1:
                        self.sn.data[ blockname ][lb:ub] = np.fromfile( f, dtype=endian+blocktype, count=elements )
                    else:
                        self.sn.data[ blockname ][lb:ub,:] = np.fromfile( f, dtype=endian+blocktype, count=elements ).reshape( npartptype, dim )

                    nsum += self.sn.nparticlesall[ptype]
                    
            nparttot += self.sn.NumPart_ThisFile
        
        if self.filecount > 1:
            self.sn.npart = self.sn.npartall

        self.sn.npart_loaded = self.sn.nparticlesall
        
        if self.sn.__verbose__:
            print('%d particles loaded.' % self.sn.npartall)

    def close(self):
        pass
    
    def write(self):
        raise Exception("Writing of format 2 files is not supported")
    
    def get_block_size_from_table( self, block ):
        npart = 0
        npartall = 0
        if block in ["POS ", "VEL ", "ID  ", "POT ", "TSTP"]:
            # present for all particles
            npart = self.sn.NumPart_ThisFile.sum()
            npartall = self.sn.nparticlesall.sum()
        elif block in ["U   ","RHO ", "NE  ", "NH  ", "HSML", "SFR ", "Z   ", "XNUC", "PRES", "VORT", "VOL ", "HRGM", "REF ", "DIVV", "ROTV", "DUDT", "BFLD", "DIVB", "PSI ", "REF ", "PASS", "TEMP", "GRAR", "GRAP", "NONN", "CSND"]:
            # present for hydro particles
            npart += self.sn.NumPart_ThisFile[0]
            npartall += self.sn.nparticlesall[0]
        elif block in ["AGE ", "Z   "]:
            # present for star particles
            npart += self.sn.NumPart_ThisFile[4]
            npartall += self.sn.nparticlesall[4]
        elif block in ["BHMA", "BHMD"]:
            #present for black hole particles
            npart += self.sn.NumPart_ThisFile[5]
            npartall += self.sn.nparticlesall[5]  
        elif block in ["MASS"]:
            npart = self.sn.NumPart_ThisFile[np.where(self.sn.MassTable==0)[0]].sum()
            npartall = self.sn.nparticlesall[np.where(self.sn.MassTable==0)[0]].sum()  
        else:
            raise Exception("Unknown block: %s. Add to format 2 loader")

        if self.tracer:
            if block in ["MASS"]:
                npart -= self.sn.NumPart_ThisFile[ self.tracer ]
                npartall -= self.sn.nparticlesall[ self.tracer ]
        return npart, npartall
    
    def parttype_has_block( self, ptype, block ):
        if block in ["POS ", "VEL ", "ID  ", "MASS", "POT ", "TSTP"]:
            if self.tracer and block in ["MASS"]:
                if ptype == self.tracer:
                    return False
            return True
        elif block in ["U   ","RHO ", "NE  ", "NH  ", "HSML", "SFR ", "Z   ", "XNUC", "PRES", "VORT", "VOL ", "HRGM", "REF ", "DIVV", "ROTV", "DUDT", "BFLD", "DIVB", "PSI ","MASS", "PASS", "TEMP", "ACCE", "P   ", "GRAR", "GRAP", "NONN", "CSND"]:
            if ptype == 0:
                return True
            else:
                return False
        elif block in ["AGE ", "Z   "]:
            if ptype == 4:
                return True
            else:
                return False
        elif block in ["BHMA", "BHMD"]:
            if ptype == 5:
                return True
            else:
                return False
        else:
            raise Exception("Unknown block: %s. Add to format 2 loader")
        
    def get_block_dim_from_table( self, block ):
        if block in ["POS ", "VEL ", "DIVV", "VORT", "BFLD", "GRAR", "GRAP"]:
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
        if self.sn.__verbose__:
            print("File %s is empty." % filename)

    return (endian_local != endian_data, endian_data)
