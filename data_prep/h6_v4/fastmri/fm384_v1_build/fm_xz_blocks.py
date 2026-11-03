"""Original-block CRC64 wrapper/index decoder extracted from prior official-source route. No old converter, pins, seek plan or defaults invoked."""
import bisect,io,lzma,zlib,hashlib
from concurrent.futures import ThreadPoolExecutor

def encode_vli(n):
    out=bytearray()
    while n>=128:
        out.append((n&127)|128);n>>=7
    out.append(n)
    return bytes(out)

def block_stream(meta,record,data):
    """Wrap one ORIGINAL block, including its CRC64, in a valid XZ stream."""
    if len(data)!=record['padded_size']:raise ValueError('Truncated original XZ block')
    index=b'\0'+encode_vli(1)+encode_vli(record['unpadded_size'])+encode_vli(record['uncompressed_size'])
    index+=b'\0'*((-len(index))%4)
    index+=zlib.crc32(index).to_bytes(4,'little')
    tail=(len(index)//4-1).to_bytes(4,'little')+bytes.fromhex(meta['flags_hex'])
    return bytes.fromhex(meta['header_hex'])+data+index+zlib.crc32(tail).to_bytes(4,'little')+tail+b'YZ'

def parse_index(header,index,footer,size):
    if (len(header)!=12 or len(footer)!=12 or header[:6]!=b'\xfd7zXZ\x00' or
        footer[-2:]!=b'YZ' or header[6:8]!=footer[8:10] or
        zlib.crc32(header[6:8])!=int.from_bytes(header[8:12],'little') or
        zlib.crc32(footer[4:10])!=int.from_bytes(footer[:4],'little') or
        (int.from_bytes(footer[4:8],'little')+1)*4!=len(index) or
        index[0]!=0 or zlib.crc32(index[:-4])!=int.from_bytes(index[-4:],'little')):
        raise ValueError('XZ metadata CRC/structure mismatch')
    if (header[7]&15)!=4:raise ValueError('Expected original CRC64 XZ blocks')
    pos=1
    def vli():
        nonlocal pos
        n=0
        for i in range(9):
            c=index[pos];pos+=1;n|=(c&127)<<(7*i)
            if c<128:
                if i and c==0:raise ValueError('Noncanonical XZ VLI')
                return n
        raise ValueError('Invalid XZ VLI')
    count=vli();records=[];coff=12;uoff=0
    if not 1<=count<=1000000:raise ValueError('Invalid XZ block count')
    for _ in range(count):
        unpadded,usize=vli(),vli();padded=(unpadded+3)//4*4
        if not 0<usize<=64*1024**2 or not 0<unpadded<=64*1024**2:
            raise ValueError('XZ block exceeds bounded memory policy')
        records.append(dict(compressed_offset=coff,unpadded_size=unpadded,padded_size=padded,
                            uncompressed_offset=uoff,uncompressed_size=usize))
        coff+=padded;uoff+=usize
    if set(index[pos:-4])-{0} or coff+len(index)+12!=size:
        raise ValueError('XZ index coverage mismatch')
    return dict(archive_bytes=size,blocks=records,block_count=count,uncompressed_bytes=uoff,
                header_hex=header.hex(),flags_hex=header[6:8].hex(),check_id=4,
                index_sha256=hashlib.sha256(index).hexdigest(),index_start=coff,
                metadata_crc_verified=True)

class IndexedDecoded(io.RawIOBase):
    def __init__(self,source,meta,start_offset,workers=1,callback=None):
        super().__init__()
        assert 1<=workers<=2, "EXPLICIT_SHARED_HTTP_WORKERS_1_OR_2"
        self.source,self.meta,self.callback=source,meta,callback
        self.start_offset=start_offset
        self.first=bisect.bisect_right([r['uncompressed_offset'] for r in meta['blocks']],start_offset)-1
        if not 0<=self.first<len(meta['blocks']):raise ValueError('Resume offset outside XZ blocks')
        self.skip=start_offset-meta['blocks'][self.first]['uncompressed_offset']
        self.next=self.first;self.schedule=self.first;self.futures={};self.pending=memoryview(b'')
        self.records=[];self.consumed_compressed_end=meta['blocks'][self.first]['compressed_offset']
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='fastmri-xz')
        self.workers=workers
        for _ in range(workers):self._schedule()

    def _schedule(self):
        if self.schedule<len(self.meta['blocks']):
            i=self.schedule;self.futures[i]=self.pool.submit(self._decode,i);self.schedule+=1

    def _decode(self,i):
        r=self.meta['blocks'][i]
        compressed=self.source.fetch(r['compressed_offset'],r['compressed_offset']+r['padded_size']-1)
        decoder=lzma.LZMADecompressor(memlimit=128*1024**2)
        raw=decoder.decompress(block_stream(self.meta,r,compressed),max_length=r['uncompressed_size']+1)
        if len(raw)!=r['uncompressed_size'] or not decoder.eof or decoder.unused_data:
            raise ValueError('XZ block decoded size/CRC completion mismatch')
        return raw,dict(block_index=i,compressed_offset=r['compressed_offset'],padded_size=r['padded_size'],
                        uncompressed_size=len(raw),compressed_sha256=hashlib.sha256(compressed).hexdigest(),
                        original_crc64_verified=True)

    def readable(self):return True

    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)

    def read(self,size=-1):
        if size<0:size=1024*1024
        out=bytearray()
        while len(out)<size:
            if not self.pending:
                if self.next>=len(self.meta['blocks']):break
                raw,record=self.futures.pop(self.next).result()
                self.records.append(record);self.next+=1;self._schedule()
                self.consumed_compressed_end=record['compressed_offset']+record['padded_size']
                self.pending=memoryview(raw)[self.skip:];self.skip=0
                if self.callback:self.callback()
            n=min(size-len(out),len(self.pending));out.extend(self.pending[:n]);self.pending=self.pending[n:]
        return bytes(out)

    def close(self):
        if not self.closed:
            self.source.stop.set();self.pool.shutdown(wait=True,cancel_futures=True)
        super().close()
