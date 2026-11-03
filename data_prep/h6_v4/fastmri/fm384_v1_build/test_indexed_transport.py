import hashlib,http.client,json,lzma,subprocess,unittest
from unittest import mock
from fm_xz_blocks import parse_index,IndexedDecoded
from fm_indexed_source import Source
class MemorySource:
 def __init__(self,data):
  import threading
  self.data=data;self.stop=threading.Event()
 def fetch(self,a,b):return self.data[a:b+1]
class IndexedTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.raw=bytes(range(256))*12288;cls.data=subprocess.run(['xz','--format=xz','--check=crc64','--threads=1','--block-size=1MiB','--stdout'],input=cls.raw,capture_output=True,check=True).stdout;f=cls.data[-12:];n=(int.from_bytes(f[4:8],'little')+1)*4;cls.meta=parse_index(cls.data[:12],cls.data[-12-n:-12],f,len(cls.data))
 def test_suffix_exact_offset(self):
  src=MemorySource(self.data);d=IndexedDecoded(src,self.meta,1024**2+123,workers=2)
  try:
   out=bytearray()
   while True:
    b=d.read(314159)
    if not b:break
    out.extend(b)
   self.assertEqual(bytes(out),self.raw[1024**2+123:]);self.assertEqual([r['block_index'] for r in d.records],[1,2]);self.assertTrue(all(r['original_crc64_verified'] for r in d.records))
  finally:d.close()
 def test_original_middle_CRC_failure(self):
  data=bytearray(self.data);r=self.meta['blocks'][1];data[r['compressed_offset']+r['unpadded_size']-8]^=1;d=IndexedDecoded(MemorySource(data),self.meta,1024**2,workers=1)
  try:
   with self.assertRaises(lzma.LZMAError):d.read(1)
  finally:d.close()
 def test_index_CRC_failure(self):
  f=self.data[-12:];n=(int.from_bytes(f[4:8],'little')+1)*4;i=bytearray(self.data[-12-n:-12]);i[-1]^=1
  with self.assertRaises(ValueError):parse_index(self.data[:12],i,f,len(self.data))
 def test_worker_bound(self):
  with self.assertRaises(AssertionError):IndexedDecoded(MemorySource(self.data),self.meta,0,workers=3)
 def test_ETag_length_version_reject(self):
  class Response:
   status=206
   headers={'Content-Range':'bytes 0-0/2','ETag':'wrong','x-amz-version-id':None}
   def __enter__(self):return self
   def __exit__(self,*a):pass
  class Connection:
   sock=None
   def request(self,*a,**kw):pass
   def getresponse(self):return Response()
   def close(self):pass
  s=Source('https://fastmri-dataset.s3.amazonaws.com/test?synthetic=1',{'path':'/test','bytes':2,'etag':'expected','version_id':None})
  with mock.patch.object(s,'connection',return_value=Connection()):
   with self.assertRaisesRegex(AssertionError,'ETAG_VERSION_MISMATCH'):s.fetch(0,0)
if __name__=='__main__':unittest.main()
