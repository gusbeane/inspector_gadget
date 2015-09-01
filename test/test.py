import pytest
import gadget
import numpy as np


N = 16**3

@pytest.fixture(scope="module")
def create_snapshot():
    ics = gadget.ICs("snapshot_000.0.hdf5", [N,N,0,0,0,0])
    ics.NumFilesPerSnapshot = 2
    ics.NumPart_Total[:] = 2*ics.NumPart_ThisFile
    ics.part1.pos[...] = np.random.rand(N,3)
    ics.BoxSize = 1.
    ics.write()
    
    ics = gadget.ICs("snapshot_000.1.hdf5", [N,N,0,0,0,0])
    ics.NumFilesPerSnapshot = 2
    ics.NumPart_Total[:] = 2*ics.NumPart_ThisFile
    ics.part1.pos[...] = np.random.rand(N,3)
    ics.BoxSize = 1.
    ics.write()
    
    ics = gadget.ICs("snapshot_001.0.hdf5", [N,N,0,0,0,0])
    ics.NumFilesPerSnapshot = 2
    ics.NumPart_Total[:] = 2*ics.NumPart_ThisFile
    ics.part1.pos[...] = np.random.rand(N,3)
    ics.BoxSize = 1.
    ics.write()

    ics = gadget.ICs("snapshot_001.1.hdf5", [N,N,0,0,0,0])
    ics.NumFilesPerSnapshot = 2
    ics.NumPart_Total[:] = 2*ics.NumPart_ThisFile
    ics.part1.pos[...] = np.random.rand(N,3)
    ics.BoxSize = 1.
    ics.write()
    
    return "snapshot_000.0.hdf5"
    
@pytest.fixture(scope="module")
def snapshot(create_snapshot):
    sn = gadget.Snapshot(create_snapshot)
    return sn
    
@pytest.fixture(scope="module")
def simulation(create_snapshot):
    sn = gadget.Simulation(create_snapshot)
    return sn
    
def test_open1(create_snapshot):
    sn = gadget.Snapshot(create_snapshot)
    sn.close()
    
    
def test_open2(create_snapshot):
    sn = gadget.Snapshot(".",0)
    sn.close()
    sn = gadget.Snapshot(".",1)
    sn.close()
    
def test_iter(create_snapshot):
    sn = gadget.Snapshot(".",0)
    assert(sn.filenum==0)
    i = 0
    for s in sn.iterFiles():
        assert(s.NumPart_ThisFile[0]==N)
        assert(sn.filenum==i)
        sn.part0.pos
        i = i + 1
    assert(i==2)
    s.close()
    
def test_iter2(create_snapshot):
    sn = gadget.Snapshot(".",0)
    assert(sn.snapshot==0)
    assert(sn.nextSnapshot()==True)
    assert(sn.snapshot==1)
    sn.part0.pos
    sn.close()
    
    
def test_load_combine(create_snapshot):
    sn = gadget.Snapshot(".",0, combineFiles=True)
    assert(sn.npart_loaded[0] == sn.header.NumPart_Total[0])
    sn.close()
    
def test_load_fields(create_snapshot):
    sn = gadget.Snapshot(".",0, fields=['pos','Velocities'])
    sn.pos
    assert(hasattr(sn,"id")==False)
    assert(hasattr(sn,"vel")==True)
    assert(hasattr(sn,"pos")==True)
    sn.close()
    
def test_load_todouble(create_snapshot):
    sn = gadget.Snapshot(".",0)
    assert(sn.pos.dtype==np.float32)
    assert(sn._precision==np.float32)
    sn.close()
    
    sn = gadget.Snapshot(".",0, toDouble=True)
    assert(sn.pos.dtype==np.float64)
    assert(sn._precision==np.float64)
    sn.close()
    
    
def test_particles(create_snapshot):
    sn = gadget.Snapshot(".",0, parttype=[0])
    sn.part0.pos
    assert(sn.npart_loaded.sum() == sn.NumPart_ThisFile[0])
    assert(sn.npart_loaded[1] == 0)
    
def test_access(snapshot):
    snapshot.pos
    snapshot["pos"]
    snapshot["Coordinates"]
    snapshot.pos_0
    snapshot.pos_x
    
    assert( (snapshot.pos == snapshot.Coordinates).all() )
    
def test_access2(snapshot):
    snapshot.part0.pos
    snapshot.part0["pos"]
    snapshot.part0["Coordinates"]
    snapshot.part0.pos_0
    snapshot.part0.pos_x
    assert((snapshot.pos[N:2*N,:] == snapshot.part1.pos[...]).all())
    
    
def test_write(snapshot):
    snapshot.write("test_write.hdf5")
    
    sn = gadget.Snapshot("test_write.hdf5")
    assert(sn.pos.shape[0] == 2*N)
    assert(sn.part0.pos.shape[0] == N)
    sn.close()
    
def test_header(snapshot):
    snapshot.header
    assert((snapshot.NumPart_ThisFile == snapshot.header.NumPart_ThisFile).all())
    assert((snapshot.header.NumPart_ThisFile[:2]==N).all())
    assert((snapshot.header.NumPart_Total[:2]==2*N).all())
    assert(snapshot.header.NumFilesPerSnapshot == 2)
    
    assert((snapshot.npart_loaded == snapshot.NumPart_ThisFile).all())
    assert(snapshot.npart == snapshot.NumPart_ThisFile.sum())
    assert((snapshot.nparticlesall == snapshot.NumPart_Total).all())
    assert(snapshot.nparticlesall.sum() == snapshot.npartall)
    
def test_addfield(create_snapshot):
    sn = gadget.Snapshot(create_snapshot)
    sn.addField("blubb", pres=[3,0,0,0,0,0])
    sn.blubb
    sn.part0.blubb
    assert(sn.blubb.dtype==np.float32)
    assert(sn.blubb.shape[1]==3)
    assert(sn.blubb.shape[0]==N)
    
    sn.addField("blubb2", pres=[3,0,0,0,0,0],dtype=np.float64)
    sn.blubb2
    assert(sn.blubb2.dtype==np.float64)
    

def test_in(snapshot):
    assert("pos" in snapshot)
    assert("pos_x" in snapshot)
    assert("pos_0" in snapshot)
    assert(("pos_w" in snapshot) == False)
    
    
def test_in2(snapshot):
    assert("pos" in snapshot.part0)
    assert("pos_x" in snapshot.part0)
    assert("pos_0" in snapshot.part0)
    assert(("pos_w" in snapshot.part0) == False)
    
    
def test_print(snapshot):
    print(snapshot)
    
    
    