"""Export the RSNA readout in public Net names; full recovery weights stay intact."""
from pathlib import Path
import argparse
import torch
from common import ROOT,POLICY,read,sha,pin_check
from loop import atomic_torch


def public_rsna_state(state):
    result={}
    for name,tensor in state.items():
        if name.startswith('aux_') or name.startswith('oai_vm.'):continue
        key='enc.'+name[len('encoder.'):] if name.startswith('encoder.') else name[len('main.'):] if name.startswith('main.') else name
        if key in result:raise ValueError('EXPORT_NAME_COLLISION')
        result[key]=tensor.detach().cpu().clone()
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['core6','vm'],required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    package=pin_check();source=Path(a.checkpoint).resolve();out=(ROOT/'outputs'/a.arm).resolve()
    if not source.is_relative_to(out):raise ValueError('CHECKPOINT_OUTSIDE_ARM')
    state=torch.load(source,map_location='cpu',weights_only=False)
    if state['package_sha256']!=package or state['arm']!=a.arm:raise ValueError('EXPORT_CHECKPOINT_BINDING')
    destination=Path(a.output)
    if destination.exists():raise ValueError('EXPORT_ALREADY_EXISTS')
    atomic_torch(destination,{'model':public_rsna_state(state['model']),'arm':a.arm,'package_sha256':package,'source_checkpoint_sha256':sha(source),
        'architecture':'public spatial ConvNeXtSmall384 crop352 hidden384 syn-decomp','orientation_policy':POLICY,
        'recipe':read(ROOT/'CONFIG.json'),'readout':'RSNA_MAIN12_ONLY','auxiliary_heads_removed':True})


if __name__=='__main__':main()
