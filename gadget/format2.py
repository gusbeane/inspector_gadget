import numpy as np
import os.path as path
import pylab
import matplotlib
import struct
import time

import gadget.loader as loader
import gadget.fields as fields

class Format2:

    def __init__( self, sn, filename, verbose=False, onlyHeader=False, nommap=False, tracer=False, toDouble=False, **param):
        self.sn = sn
        if sn.__fields__ != None:
            self.loadlist = sn.__fields__
    	else:
            self.loadlist = []
            
        self.verbose = verbose
        self.nommap=nommap
        self.tracer = tracer
        self.onlyHeader = onlyHeader
        self.toDouble = toDouble
        self.filename = filename


        self.datablocks_skip = ["HEAD"]
        self.datablocks_int32 = ["ID  ","NONN"]

    def load(self):
        self.filecount = 1
        if path.exists( self.filename ):
            self.files = [self.filename]
        elif path.exists( self.filename + '.0' ):
            self.files = [self.filename + '.0']
            while path.exists( self.filename + ".%d" % self.filecount ):
                self.files += [self.filename + ".%d" % self.filecount]
                self.filecount += 1

            if not nommap:
                print "Multiple files detected, thus mmap is deactivated."
                self.nommap = True
        else:
            raise Exception( "Neither %s nor %s.0 exists." % (self.filename, self.filename) )


        self.load_header( 0, verbose=self.verbose )
        self.get_blocks( 0, verbose=self.verbose )

        if self.onlyHeader:
            return

        if self.filecount == 1 and not self.nommap and not self.toDouble:
            self.load_data_mmap()
        else:
            self.load_data()

        self.sn.header = loader.Header(self.sn)
        self.sn.part0 = loader.PartGroup(self.sn,0)
        self.sn.part1 = loader.PartGroup(self.sn,1)
        self.sn.part2 = loader.PartGroup(self.sn,2)
        self.sn.part3 = loader.PartGroup(self.sn,3)
        self.sn.part4 = loader.PartGroup(self.sn,4)
        self.sn.part5 = loader.PartGroup(self.sn,5)
        self.sn.groups = [ self.sn.part0, self.sn.part1, self.sn.part2, self.sn.part3, self.sn.part4, self.sn.part5]



    def get_blocks( self, fileid, verbose=False ):
        swap, endian = self.endianness_check( self.files[fileid] )
		
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
                    fields.isPresent(name.strip().lower(), self.sn, learn=True, gr=i, shape=self.get_block_dim_from_table(name))

            length = self.get_block_length( fileid, fpos + 16, name, length, endian, verbose=verbose )
            if length < 0:
                return False

            if verbose:
                print "Block %s, length %d, offset %d." %(name, length, fpos)

            f.seek( length, 1 )
            fpos = f.tell()
            s = f.read(16)
			
            self.origdata += [name]

        if verbose:
            print "%d blocks detected." % len(self.blocks.keys())
        return True

    def get_block_length( self, fileid, start, name, length, endian, verbose=False ):
        if verbose:
            print "Getting block length of block %s." % (name)
            
        if self.check_block_length( fileid, start, length, endian ):
            return length

        if verbose:
            print "First check failed."

        if name == "XNUC" or name == "PASS":
            bs, bsall = self.get_block_size_from_table( name )
            if self.sn.flag_doubleprecision:
                ele = 8
            else:
                ele = 4

            for dim in range( 500 ):
                length = bs * dim * ele + 8
                if self.check_block_length( fileid, start, length, endian ):
                    if verbose:
                        print "Block %s solved, bs=%d, dim=%d." % (name, bs, dim)
                    return length

            print "Error determining the length of block %s." % (name)
            return -1
			
        else:
            bs, bsall = self.get_block_size_from_table( name )
            dim = self.get_block_dim_from_table( name )
            if self.sn.flag_doubleprecision:
                ele = 8
            else:
                ele = 4

            length = bs * dim * ele + 8
			
            if self.check_block_length( fileid, start, length, endian ):
                return length
            else:
                print "Error determining the length of block %s." % (name)
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
        swap, endian = self.endianness_check( self.files[fileid] )
		
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
                self.sn.nparticles = np.longlong( struct.unpack( endian + "6i", s ) )
                s = f.read(48)
                self.sn.masses = np.array( struct.unpack( endian + "6d", s ) )
                s = f.read(24)
                self.sn.time, self.sn.redshift, self.sn.flag_sfr, self.sn.flag_feedback = struct.unpack( endian + "ddii", s )
                s = f.read(24)
                self.sn.nparticlesall = np.longlong( struct.unpack( endian + "6i", s ) )
                s = f.read(16)
                self.sn.flag_cooling, self.sn.num_files, self.sn.boxsize = struct.unpack( endian + "iid", s )
                s = f.read(24)
                self.sn.omega0, self.sn.omegalambda, self.sn.hubbleparam = struct.unpack( endian + "ddd", s )
                s = f.read(8)
                self.sn.flag_stellarage, self.sn.flag_metals = struct.unpack( endian + "ii", s )
                s = f.read(24)
                self.sn.nparticlesall += np.longlong( struct.unpack( endian + "6i", s ) )<<32
                s = f.read(12)
                self.sn.flag_entropy_instead_u, self.sn.flag_doubleprecision, self.sn.flag_lpt_ics = struct.unpack( "iii", s )
                s = f.read(52)


                if self.sn.nparticlesall.sum() == 0:
                    self.sn.nparticlesall = self.sn.nparticles
                        
                sort = self.sn.nparticlesall.argsort()
                if self.sn.nparticlesall[ sort[::-1] ][1] == 0:
                    self.sn.singleparticlespecies = True
                else:
                    self.sn.singleparticlespecies = False


                f.close()
                s = "" # stop reading

                self.sn.npart = self.sn.nparticles.sum()
                self.sn.npartall = self.sn.nparticlesall.sum()

                if verbose:
                    print "nparticlesall:", self.sn.nparticlesall, "sum:", self.sn.npartall

                if fileid == 0:
                    if (self.sn.num_files != self.filecount) and not (self.sn.num_files == 0 and self.filecount == 1):
                        raise Exception( "Number of files detected (%d) and num_files in the header (%d) are inconsistent." % (self.filecount, self.sn.num_files) )
        if verbose:
            print "Snapshot contains %d particles." % self.sn.npartall
        return

    def load_data_mmap( self ):
        # memory mapping works only if there is only one file
        swap, endian = self.endianness_check( self.files[0] )
        self.sn.data = {}

        self.get_blocks( 0, verbose=self.verbose )
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
                if self.sn.flag_doubleprecision:
                    blocktype = 'f8'
                    elementsize = 8
                else:
                    blocktype = 'f4'
                    elementsize = 4

            nsum = 0
            for ptype in range( 6 ):
                if self.sn.nparticles[ptype] == 0:
                    continue
                    
                if not self.parttype_has_block( ptype, name ):
                    continue

                nsum += self.sn.nparticlesall[ptype]
                
            elements = (length-8)/elementsize
            dim = elements / nsum
            
            if self.verbose:
                print "Loading block %s, offset %d, length %d, elements %d, dimension %d, particles %d/%d." % (block, self.blocks[block], length, elements, dim, nsum, self.sn.nparticlesall.sum())

            offset = f.tell() + 4
            blockname = block.strip().lower()
            if dim == 1:
                self.sn.data[ blockname ] = np.memmap( self.files[0], offset=offset, mode='c', dtype=endian+blocktype, shape=(nsum) )
            else:
                self.sn.data[ blockname ] = np.memmap( self.files[0], offset=offset, mode='c', dtype=endian+blocktype, shape=(nsum, dim) )

        self.sn.npart_loaded = self.sn.nparticlesall
        return

    def load_data( self ):
        swap, endian = self.endianness_check( self.files[0] )
        self.sn.data = {}

        nparttot = pylab.zeros( 6, dtype='int32' )    
        for fileid in range( self.filecount ):
            self.load_header( fileid, verbose=self.verbose )
            self.get_blocks( fileid, verbose=self.verbose )
            
            f = open( self.files[fileid], 'r' )
            for block in self.blocks.keys():
                # skip some blocks
                if block in self.datablocks_skip:
                    continue

                if len(self.loadlist) > 0 and not block.strip().lower() in self.loadlist:
                    continue

                if self.verbose:
                    print "Loading block %s of file %s." % (block, fileid)

                f.seek( self.blocks[block], 0 )
                fheader, name, length, ffooter = struct.unpack( endian + "i4sii", f.read(16) )
                length = self.get_block_length( fileid, f.tell(), name, length, endian )
                f.seek( 4, 1 ) # skip fortran header of data field

                if block in self.datablocks_int32:
                    blocktype = "i4"
                    elementsize = 4
                else:
                    if self.sn.flag_doubleprecision:
                        blocktype = 'f8'
                        elementsize = 8
                    else:
                        blocktype = 'f4'
                        elementsize = 4
                
                nsum = 0
                for ptype in range( 6 ):
                    if self.sn.nparticles[ptype] == 0:
                        continue
                    
                    if not self.parttype_has_block( ptype, name ):
                        continue
                    
                    npart, npartall = self.get_block_size_from_table( name )
                    dim = self.get_block_dim_from_table( name )
                    npartptype = self.sn.nparticles[ptype]
                    elements = npartptype * dim
                    
                    if self.verbose:
                        print "Loading block %s, offset %d, length %d, elements %d, dimension %d, type %d, particles %d/%d." % (block, self.blocks[block], length, elements, dim, ptype, npartptype, npartall)
                    
                    blockname = block.strip().lower()
                    if not self.sn.data.has_key( blockname ):
                        if dim == 1:
                            self.sn.data[ blockname ] = pylab.zeros( npartall, dtype=blocktype )
                        else:
                            self.sn.data[ blockname ] = pylab.zeros( (npartall, dim), dtype=blocktype )

                    if blocktype == 'f4' and self.toDouble:
                        self.sn.data[ blockname ] = self.sn.data[ blockname ].astype( 'float64' ) # change array type to float64

                    lb = nsum + nparttot[ptype]
                    ub = lb + npartptype
                    
                    if self.verbose:
                        print "Block contains %d elements (length=%g, elementsize=%g, lb=%d, ub=%d)." % (elements,length,elementsize,lb,ub)

                    if dim == 1:
                        self.sn.data[ blockname ][lb:ub] = np.fromfile( f, dtype=endian+blocktype, count=elements )
                    else:
                        self.sn.data[ blockname ][lb:ub,:] = np.fromfile( f, dtype=endian+blocktype, count=elements ).reshape( npartptype, dim )

                    nsum += self.sn.nparticlesall[ptype]
                    
            nparttot += self.sn.nparticles
        
        if self.filecount > 1:
            self.sn.npart = self.npartall
        print '%d particles loaded.' % self.sn.npartall
        self.sn.npart_loaded = self.sn.nparticlesall
        return

    def get_block_size_from_table( self, block ):
        npart = 0
        npartall = 0
        if block in ["POS ", "VEL ", "ID  ", "MASS", "POT ", "TSTP"]:
            # present for all particles
            npart = self.sn.nparticles.sum()
            npartall = self.sn.nparticlesall.sum()
        if block in ["U   ","RHO ", "NE  ", "NH  ", "HSML", "SFR ", "Z   ", "XNUC", "PRES", "VORT", "VOL ", "HRGM", "REF ", "DIVV", "ROTV", "DUDT", "BFLD", "DIVB", "PSI ", "MASS", "REF ", "PASS", "TEMP", "GRAR", "GRAP", "NONN", "CSND"]:
            # present for hydro particles
            npart += self.sn.nparticles[0]
            npartall += self.sn.nparticlesall[0]
        if block in ["AGE ", "Z   "]:
            # present for star particles
            npart += self.sn.nparticles[4]
            npartall += self.sn.nparticlesall[4]
        if block in ["BHMA", "BHMD"]:
            #present for black hole particles
            npart += self.sn.nparticles[5]
            npartall += self.sn.nparticlesall[5]

        if self.tracer:
            if block in ["MASS"]:
                npart -= self.sn.nparticles[ self.tracer ]
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
                return False
        
    def get_block_dim_from_table( self, block ):
        if block in ["POS ", "VEL ", "DIVV", "VORT", "BFLD", "GRAR", "GRAP"]:
            return 3
        else:
            return 1

    def endianness_local(self):
        s = struct.pack( "bb", 1, 0 )
        r, = struct.unpack( "h", s )

        if r == 1:
            return '<'
        else:
            return '>'

    def endianness_check(self, filename ):
        if not path.exists( filename ):
            print "File %s does not exist." % filename
            return False

        filesize = path.getsize( filename )

        endian_local = self.endianness_local()
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
        			print "File %s is corrupt." % filename
        			return False
        else:
        	f.close()
        	print "File %s is empty." % filename

        return (endian_local != endian_data, endian_data)
