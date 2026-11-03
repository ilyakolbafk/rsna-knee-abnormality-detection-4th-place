"""Official version-pinned block ranges; per-socket MSS, no host network edits."""
import hashlib,http.client,json,os,socket,ssl,threading,time,urllib.parse
from pathlib import Path
from fm_xz_blocks import parse_index
HOST='fastmri-dataset.s3.amazonaws.com'
class Source:
 def __init__(self,url,identity,callback=None,mss=1200):
  u=urllib.parse.urlsplit(url);assert u.scheme=='https' and u.hostname==HOST and not u.fragment and u.path==identity['path']
  self.url=url;self.path=u.path+'?'+u.query;self.identity=identity;self.size=identity['bytes'];self.etag=identity['etag'];self.callback=callback;self.mss=mss;self.stop=threading.Event();self.lock=threading.Lock();self.wire_bytes=0;self.connections=set()
 def connection(self):
  c=http.client.HTTPSConnection(HOST,timeout=15,context=ssl.create_default_context())
  def create(address,timeout=15,source_address=None):
   sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_MAXSEG,self.mss);sock.settimeout(timeout)
   try:sock.connect((socket.gethostbyname(HOST),443));return sock
   except BaseException:sock.close();raise
  c._create_connection=create
  with self.lock:self.connections.add(c)
  return c
 def fetch(self,start,end):
  assert 0<=start<=end<self.size
  for attempt in range(4):
   if self.stop.is_set():raise RuntimeError('INDEXED_SOURCE_CANCELLED')
   conn=None
   try:
    conn=self.connection();conn.request('GET',self.path,headers={'Range':'bytes=%d-%d'%(start,end),'If-Match':self.etag,'Accept-Encoding':'identity'})
    with conn.getresponse() as response:
     if response.status!=206:raise ValueError('HTTP_STATUS_%d_URL_REDACTED'%response.status)
     assert response.headers.get('Content-Range')=='bytes %d-%d/%d'%(start,end,self.size),'RANGE_VERSION_MISMATCH'
     assert response.headers.get('ETag')==self.etag and response.headers.get('x-amz-version-id')==self.identity['version_id'],'ETAG_VERSION_MISMATCH'
     result=bytearray();began=time.monotonic()
     while len(result)<end-start+1:
      if self.stop.is_set():raise RuntimeError('INDEXED_SOURCE_CANCELLED')
      if time.monotonic()-began>120:raise OSError('BOUNDED_RANGE_TIMEOUT')
      b=response.read(min(65536,end-start+1-len(result)))
      if not b:raise OSError('TRUNCATED_OFFICIAL_RANGE')
      result.extend(b)
      with self.lock:self.wire_bytes+=len(b)
      if self.callback:self.callback(self)
     assert response.read(1)==b'','EXTRA_RANGE_BODY_BYTES'
     return bytes(result)
   except (OSError,http.client.IncompleteRead):
    if attempt==3:raise RuntimeError('TRANSIENT_RANGE_RETRY_EXHAUSTED_URL_REDACTED') from None
    if self.stop.wait(min(8,2**attempt)):raise RuntimeError('INDEXED_SOURCE_CANCELLED') from None
   finally:
    if conn:
     conn.close()
     with self.lock:self.connections.discard(conn)
 def index(self):
  footer=self.fetch(self.size-12,self.size-1);length=(int.from_bytes(footer[4:8],'little')+1)*4
  assert 8<=length<=16*1024**2,'UNBOUNDED_INDEX'
  header=self.fetch(0,11);index=self.fetch(self.size-12-length,self.size-13);meta=parse_index(header,index,footer,self.size)
  assert header[6]==0 and header[7]==4,'UNEXPECTED_XZ_STREAM_FLAGS'
  return meta,{'header':header,'index':index,'footer':footer}
 def close(self):
  self.stop.set()
  with self.lock:connections=list(self.connections)
  for c in connections:
   if c.sock:
    try:c.sock.shutdown(socket.SHUT_RDWR)
    except OSError:pass
   c.close()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
