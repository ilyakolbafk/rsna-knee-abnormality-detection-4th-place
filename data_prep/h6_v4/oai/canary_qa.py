from common import *
from reader import CaseReader,window_rows
import argparse,math,shutil,tempfile
import numpy as np
from PIL import Image,ImageDraw

def qa(case):
    reader=CaseReader(RUN/'cache');px,meta,a=reader.read(case)
    if sha(RUN/'cache'/f'{case}.npz')!=a['sha256']:raise ValueError('CACHE_SHA_CONFLICT')
    if hashlib.sha256(px.tobytes()).hexdigest()!=a['pixel_sha256']:raise ValueError('PIXEL_HASH_CONFLICT')
    if hashlib.sha256(meta.tobytes()).hexdigest()!=a['metadata_sha256']:raise ValueError('META_HASH_CONFLICT')
    total=0;tiles=[];directions=[]
    for b in a['audit']['blocks']:
        expected=[0,1,4,5] if b['plane']!='Coronal' else [0,2,3]
        assert [x[0] for x in b['views']]==expected
        for vid,start in b['views']:
            total+=b['n_slices'];idx=start+b['n_slices']//2
            tile=Image.fromarray(px[idx]).convert('RGB');draw=ImageDraw.Draw(tile)
            g=a['audit']['view_geometry'][idx];aff=np.asarray(g['output_to_lps'],float)
            def xy(point):return np.linalg.lstsq(aff[:,:2],np.asarray(point)-aff[:,2],rcond=None)[0]
            for name,col in [('joint_lps','lime'),('patella_lps','orange')]:
                point=a['anchor'].get(name)
                if point is not None:
                    x,y=xy(point);draw.ellipse((x-5,y-5,x+5,y+5),outline=col,width=2)
                    if 0<=x<384 and 0<=y<384:draw.text((x+6,y),name.split('_')[0],fill=col)
            center=aff@np.array([191.5,191.5,1])
            for name,offset in [('A',[0,-15,0]),('P',[0,15,0]),('S',[0,0,15]),('L',[15,0,0])]:
                delta=xy(center+offset)-xy(center)
                if np.linalg.norm(delta)>5:
                    base=np.array([45.,70.]);end=base+delta
                    draw.line([tuple(base),tuple(end)],fill='cyan',width=2);draw.text(tuple(end),name,fill='cyan')
            draw.rectangle((0,0,383,35),fill='black');draw.text((4,4),f"{case} {b['family']} view={vid} slab={b['n_slices']//2}/{b['n_slices']}",fill='white')
            draw.text((4,18),f"FOV={g['fov_mm'][0]:.1f}x{g['fov_mm'][1]:.1f}mm",fill='white')
            tiles.append(tile.resize((256,256)))
            point=g['anchor_lps'];joint=np.asarray(a['anchor']['joint_lps'])
            if point is not None:
                diff=np.asarray(point)-joint
                if vid==5:assert diff[1]>0, 'POSTERIOR_LPS_DIRECTION'
                if vid==4:assert diff[1]<0, 'PF_LPS_DIRECTION'
                if b['plane']=='Coronal':
                    expected_sign=(-1 if a['decision']['side']=='L' else 1)*(1 if vid==2 else -1)
                    assert expected_sign*diff[0]>0,'MEDIAL_LATERAL_LPS_DIRECTION'
                directions.append({'family':b['family'],'view_id':vid,'offset_lps':diff.tolist()})
    assert total==len(px)
    for mode in ['train','eval']:
        images,m,rows=reader.windows(case,mode=mode)
        assert images.dtype==np.uint8 and images.shape[1:]==(3,384,384) and m.shape[1]==6
        blocks=[{**b,'views':[(0,v,start) for v,start in b['views']]} for b in a['audit']['blocks']]
        rng=np.random.default_rng(20261021) if mode=='train' else None
        cand=window_rows(blocks,lambda f,r:meta[r],mode,3,1,3,0.,rng,.25,'cover')
        if mode=='train' and len(cand)>176:cand=[cand[i] for i in np.sort(rng.choice(len(cand),176,replace=False))]
        reference=np.asarray(cand,np.int64)
        assert np.array_equal(rows,reference)
        assert np.array_equal(images,np.stack([px[reference[:,2+d]] for d in range(3)],1))
        assert np.array_equal(m,meta[reference[:,1]][:,[1,3,4,5,6,7]])
    montage=Image.new('RGB',(4*256,math.ceil(len(tiles)/4)*256),'black')
    for i,t in enumerate(tiles):montage.paste(t,((i%4)*256,(i//4)*256))
    out=RUN/'qa';out.mkdir(exist_ok=True);montage.save(out/f'{case}.montage.jpg',quality=90)
    result={'case_id':case,'status':'PASS_NUMERIC_READER_AND_LPS_DIRECTIONS','rows':len(px),'views':len(tiles),
            'pixel_metadata_bitexact':True,'train_cover_and_eval_windows_bitexact':True,
            'directions':directions,'montage':str(out/f'{case}.montage.jpg'),
            'visual_anatomy_review':'PENDING','private_hrepro_parity':False}
    dump(out/f'{case}.json',result);print(json.dumps({k:v for k,v in result.items() if k!='directions'}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cases',nargs='+');a=ap.parse_args()
    for case in a.cases:qa(case)
