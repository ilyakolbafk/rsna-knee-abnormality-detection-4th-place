"""Meaningful edge cases; real DICOM canaries are a separate build/QA command."""
from support import *
import unittest,tempfile
import numpy as np
import runtime_core as C
from cache_store import ChunkWriter,CacheReader

class RuntimeTests(unittest.TestCase):
 def test_original_model_strict_load_and_forward(self):
  model,_=C.resources()
  with C.torch.no_grad():y=model(C.torch.zeros(1,1,32,32))
  self.assertEqual(tuple(y.shape),(1,4,32,32));self.assertTrue(C.torch.isfinite(y).all().item())
 def test_slab_tail_and_short_series(self):
  self.assertEqual(len(C.slab_chunks(9,.5)),9)
  self.assertEqual([len(c)for c in C.slab_chunks(17,.5)],[7,7,3])
  self.assertEqual(len(C.slab_chunks(1,.5)),1)
 def test_non_square_D4_affines(self):
  for plane,iop in [('Sagittal',[0,-1,0,0,0,-1]),('Coronal',[1,0,0,0,0,-1]),('Axial',[1,0,0,0,1,0])]:
   for side in ['L','R']:
    n,h,w=3,17,23;v=np.arange(n*h*w,dtype=np.float32).reshape(n,h,w);tags={'ImageOrientationPatient':iop,'ImagePositionPatient':[0,0,0],'PixelSpacing':[.6,.4],'Rows':h,'Columns':w}
    aff=np.stack([np.asarray(iop[:3])*.4,np.asarray(iop[3:])*.6,np.zeros(3)],axis=1)
    s={'volume':v,'sid':'synthetic','frames':[{'tags':tags,'sop_uid':str(i)}for i in range(n)],'plane':plane,'aff':np.repeat(aff[None],n,0),'fatsat':0}
    d={'side':side,'bilateral':False,'series':[{'sid':'synthetic','plane':plane,'appearance_side':side,'apply_pose':None,'reverse_depth_extra':False}]}
    out=C.orient(s,d,side);self.assertEqual(set(out['volume'].ravel()),set(C.quantize_series(v)[0].ravel()))
    pull=np.asarray(out['pull_stored_to_original']);self.assertTrue(np.allclose(out['aff'][0],aff@pull))
 def test_original_decode_and_quantization(self):
  import pydicom
  from pydicom.uid import ExplicitVRLittleEndian
  d=pydicom.Dataset();d.file_meta=pydicom.Dataset();d.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
  d.Rows=2;d.Columns=3;d.SamplesPerPixel=1;d.PhotometricInterpretation='MONOCHROME1';d.BitsAllocated=16;d.BitsStored=16;d.HighBit=15;d.PixelRepresentation=0;d.PixelSpacing=[.5,.6];d.RescaleSlope=2;d.RescaleIntercept=-10
  raw=np.arange(6,dtype=np.uint16).reshape(2,3);d.PixelData=raw.tobytes();im=C.pixels_from_dataset(d).image
  self.assertTrue(np.array_equal(im,-10-(raw.astype(np.float32)*2-10)))
  q,info=C.quantize_series(np.full((2,3,4),7,np.float32));self.assertEqual(q.dtype,np.uint16);self.assertFalse(q.any())
 def test_resume_requires_QA_and_exact_source_inventory(self):
  from build_px_v4 import source_unchanged
  with tempfile.TemporaryDirectory()as tmp:
   folder=Path(tmp)/'cache';raw=Path(tmp)/'raw';raw.mkdir();(raw/'1.dcm').write_bytes(b'fake');source=raw/'1.dcm'
   w=ChunkWriter(folder,{'recipe':'test'},max_rows=2);w.add(np.zeros((384,384),np.uint8),[0,1,0,0,0,0,1,1,1,0,1,0]);idx=w.finish({'blocks':[{'n_slices':1,'views':[[0,0]]}],'source_files':[{'path':str(source),'stat':stat_id(source)}],'source_directory_inventory':{str(raw):['1.dcm']}})
   self.assertFalse(source_unchanged(folder,{'recipe':'test'}))
   qa=CacheReader(folder).verify();qa['verified_index_sha256']=sha(folder/'index.json');dump(folder/'QA.json',qa)
   self.assertTrue(source_unchanged(folder,{'recipe':'test'}))
   (raw/'2.dcm').write_bytes(b'new');self.assertFalse(source_unchanged(folder,{'recipe':'test'}))
 def test_chunks_physical_windows_corruption(self):
  with tempfile.TemporaryDirectory()as tmp:
   w=ChunkWriter(tmp,{'recipe':'test'},max_rows=2)
   for i in range(5):w.add(np.full((384,384),i,np.uint8),[0,1,0,0,0,np.log1p(.8),1,1,1,i,5,0])
   idx=w.finish({'blocks':[{'n_slices':5,'views':[[0,0]]}]});r=CacheReader(tmp,max_bytes=400000)
   self.assertEqual(r.verify()['rows'],5);self.assertEqual(len(idx['chunks']),3)
   for px,_,rows in r.iter_windows('eval'):
    for j,row in enumerate(rows):self.assertEqual(px[j,:,0,0].tolist(),row[2:].tolist())
   path=Path(tmp)/idx['chunks'][0]['path'];raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
   with self.assertRaises(ValueError):CacheReader(tmp).verify()

if __name__=='__main__':unittest.main()
