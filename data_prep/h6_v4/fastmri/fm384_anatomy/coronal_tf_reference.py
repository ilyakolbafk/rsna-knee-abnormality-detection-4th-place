"""Keras CPU reference, without custom executable objects or training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2')
import json,time
from pathlib import Path
import numpy as np
import h5py
import tensorflow as tf
ROOT=Path('/workspace/knee_v4_cache_20261021/fastmri_v1/anatomy_v1')
tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
model_path=ROOT/'weights/coronal_best_model.h5'
with h5py.File(model_path,'r') as f:
 cfg=f.attrs['model_config'];cfg=json.loads(cfg.decode() if isinstance(cfg,bytes) else cfg)
# Old Keras stores a singleton BN axis list; modern Keras expects the same axis
# as an integer. This changes JSON syntax only, not the operator or weights.
for layer in cfg['config']['layers']:
 if layer['class_name']=='BatchNormalization' and isinstance(layer['config']['axis'],list):
  assert len(layer['config']['axis'])==1;layer['config']['axis']=layer['config']['axis'][0]
 if layer['class_name']=='Conv2DTranspose' and 'groups' in layer['config']:
  assert layer['config'].pop('groups')==1
model=tf.keras.Model.from_config(cfg['config']);model.load_weights(model_path)
x=np.load(ROOT/'qa/coronal_parity_inputs.npy');outputs=[]
for i in range(len(x)):
 t=time.time();y=model(x[i:i+1],training=False).numpy();outputs.append(y);print(json.dumps({'TF_reference_sample':i,'seconds':time.time()-t,'shape':list(y.shape),'finite':bool(np.isfinite(y).all())}),flush=True)
np.save(ROOT/'qa/coronal_TF_reference_outputs.npy',np.concatenate(outputs));(ROOT/'qa/TF_REFERENCE_COMPLETE.json').write_text(json.dumps({'tensorflow':tf.__version__,'samples':len(x),'no_training':True,'completed_unix':time.time()},indent=2)+'\n')
