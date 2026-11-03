"""Isolated official DA3 inference with explicit depth and pixel semantics.

This worker never installs/downloads assets or grants a visual approval.
Explicit compute-only execution can prepare real source depth before M5 review.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'

if __package__ in {None, ''}:
    module_dir=Path(__file__).resolve().parent
    sys.path[:]=[entry for entry in sys.path if Path(entry or '.').resolve()!=module_dir]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from pipeline.common import read_json, ref, sha256, stable_hash, utc_now, verify_ref, write_json

OFFICIAL_API='https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/api.py'
OFFICIAL_SPEC='https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/specs.py'
OFFICIAL_METRIC='https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards'


def native_path(value: str | Path) -> Path:
    """Resolve Windows references under WSL without changing persisted paths."""
    text=str(value)
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',text):
        text='/mnt/'+text[0].lower()+'/'+text[3:].replace('\\','/')
    return Path(text)


def persisted_path(value: str | Path) -> str:
    text=str(Path(value).resolve())
    match=re.match(r'^/mnt/([a-z])/(.*)$',text)
    return (match[1].upper()+':/'+match[2]) if match else text.replace('\\','/')


def artifact_ref(value: str | Path) -> dict:
    result=ref(native_path(value));result['path']=persisted_path(native_path(value))
    return result


def checked_ref(value: dict) -> None:
    verify_ref({**value,'path':str(native_path(value['path']))})


def preflight(request: dict) -> dict:
    """No torch import, model loading, credential reads, or GPU allocation."""
    repo=native_path(request['model_repo']); model=native_path(request['model_dir'])
    if (repo/'src').is_dir():sys.path.insert(0,str(repo/'src'))
    reasons=[]
    if not (repo/'src/depth_anything_3/api.py').is_file():reasons.append('DA3_CODE_MISSING')
    if not model.is_dir() or not (model/'config.json').is_file():reasons.append('DA3_LOCAL_MODEL_MISSING')
    if not model.is_dir() or not any(model.rglob('*.safetensors')):reasons.append('DA3_WEIGHTS_MISSING')
    for dependency in ['torch','depth_anything_3']:
        try:available=importlib.util.find_spec(dependency) is not None
        except (ModuleNotFoundError,ValueError):available=False
        if not available:reasons.append('DA3_DEPENDENCY_MISSING:'+dependency)
    source=read_json(native_path(request['source_index']))
    if source.get('source_key')!=request['source_key']:reasons.append('S5_SOURCE_KEY_MISMATCH')
    frames=source.get('frames',[])
    limit=int(request.get('max_frames',52))
    if not 1<=len(frames)<=limit or limit>291:reasons.append('DA3_BOUNDED_SCOPE_INVALID')
    receipt_ref=request.get('m5_review_ref')
    review_ok=False
    if receipt_ref:
        checked_ref(receipt_ref)
        receipt=read_json(native_path(receipt_ref['path']))
        review_ok=(receipt.get('accepted') is True and receipt.get('source_key')==request['source_key']
                   and receipt.get('scope')=='episode_human_arm_removal' and bool(receipt.get('evidence')))
        if review_ok:
            for item in receipt['evidence']:
                checked_ref(item)
    if not review_ok and not request.get('compute_only',False):reasons.append('S5_M5_REVIEW_PENDING')
    return {'schema_version':'learning.stage5.da3_preflight.v1','ready':not reasons,'reasons':reasons,
            'interpreter':sys.executable,'model_repo':persisted_path(repo),'model_dir':persisted_path(model),
            'frames':len(frames),'current_cuda_execution_checked':False,'weights_loaded':False,
            'compute_only':bool(request.get('compute_only',False)),
            'm5_review_status':'accepted' if review_ok else 'pending','training_ready':False,
            'network_downloads_allowed':False,'official_api':OFFICIAL_API,'official_spec':OFFICIAL_SPEC}


def execute(request: dict, output: Path) -> dict:
    state=preflight(request)
    if not state['ready']:raise RuntimeError('DA3_PREFLIGHT_BLOCKED: '+','.join(state['reasons']))
    output=native_path(output).resolve(); source=read_json(native_path(request['source_index']))
    source_root=native_path(request['source_index']).resolve().parent
    if output==source_root or output in source_root.parents:
        raise ValueError('Output must be an independent derived directory')
    output.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    import torch
    from depth_anything_3.api import DepthAnything3
    device=request.get('device','cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA_UNAVAILABLE')
    width,height=int(source['width']),int(source['height'])
    work_width,work_height=map(int,request.get('work_size',[504,280]))
    if min(work_width,work_height)<=0 or work_width%14 or work_height%14:raise ValueError('Explicit work size must be positive multiples of 14')
    K_native=np.asarray(source['camera']['K'],np.float64)
    scale=np.diag([work_width/width,work_height/height,1.])
    K_work=scale@K_native
    if K_native.shape!=(3,3) or not np.isfinite(K_native).all() or min(K_work[0,0],K_work[1,1])<=0:
        raise ValueError('SOURCE_CAMERA_K_INVALID')
    frames=source['frames']
    times=np.asarray([row['timestamp_s'] for row in frames],np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('SOURCE_PTS_NOT_STRICTLY_INCREASING')
    indices=[int(row['frame_index']) for row in frames]
    if len(set(indices))!=len(indices) or indices!=sorted(indices):raise ValueError('SOURCE_FRAME_INDEX_INVALID')
    for row in frames:checked_ref(row)
    config=read_json(native_path(request['model_dir'])/'config.json')
    model_name=str(config.get('model_name','')).lower()
    requested_mode=request.get('depth_mode','relative')
    if requested_mode not in {'relative','metric'}:raise ValueError('Explicit depth_mode must be relative or metric')
    if requested_mode=='metric' and model_name not in {'da3metric-large','da3nested-giant-large'}:
        raise ValueError('Unsupported metric checkpoint; do not infer units from file name')
    focal_work=float((K_work[0,0]+K_work[1,1])/2)
    conversion=(focal_work/300.) if requested_mode=='metric' and model_name=='da3metric-large' else 1.
    model_files={persisted_path(p):sha256(p) for p in native_path(request['model_dir']).rglob('*') if p.is_file()}
    code_root=native_path(request['model_repo'])/'src/depth_anything_3'
    code_files={str(p.relative_to(code_root)).replace('\\','/'):sha256(p)
        for p in code_root.rglob('*') if p.is_file() and p.suffix in {'.py','.yaml','.yml'}}
    signature=stable_hash({'request':request,'source_index_sha256':sha256(native_path(request['source_index'])),
        'worker_sha256':sha256(Path(__file__)),'model_files':model_files,'official_code_files':code_files,
        'torch_version':torch.__version__,'numpy_version':np.__version__})
    final_path=output/f'scene-depth-{requested_mode}.json'
    if final_path.is_file():
        previous_manifest=read_json(final_path)
        if previous_manifest.get('provenance',{}).get('run_signature')!=signature:
            raise ValueError('DA3_OUTPUT_CONFIGURATION_MISMATCH: preserve prior results and choose a new output')
        for row in previous_manifest['frames']:
            checked_ref(row);checked_ref(row['raw_work_ref'])
            if row.get('raw_model_confidence_ref'):checked_ref(row['raw_model_confidence_ref'])
        checked_ref(previous_manifest['measurement_provenance'])
        return previous_manifest
    model=DepthAnything3.from_pretrained(str(native_path(request['model_dir']).resolve())).to(device).eval()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(request.get('seed',42)))
    result_rows=[];statistics=[];previous=None;previous_time=None;flags=[]
    # Single-view Metric-Large is deliberately streamed one frame at a time.
    # The optional nested checkpoint also records this inference context.
    for ordinal,row in enumerate(frames):
        checked_ref(row)
        checkpoint_path=output/'frame-records'/f'frame-{row["frame_index"]:06d}.json'
        if checkpoint_path.is_file():
            cached=read_json(checkpoint_path)
            if cached.get('run_signature')!=signature:
                raise ValueError('DA3_FRAME_CONFIGURATION_MISMATCH: choose a new output')
            checked_ref(cached['frame']);checked_ref(cached['frame']['raw_work_ref'])
            if cached['frame'].get('raw_model_confidence_ref'):checked_ref(cached['frame']['raw_model_confidence_ref'])
            result_rows.append(cached['frame']);statistics.append(cached['statistics']);flags.append(cached['api_is_metric'])
            previous=np.load(native_path(cached['frame']['path']),allow_pickle=False);previous_time=float(row['timestamp_s'])
            print(json.dumps({'frame':row['frame_index'],'cached':True}),flush=True)
            continue
        torch.manual_seed(int(request.get('seed',42))+int(row['frame_index']))
        image=Image.open(native_path(row['path'])).convert('RGB')
        if image.size!=(width,height):raise ValueError('SOURCE_IMAGE_SIZE_MISMATCH')
        work_image=np.asarray(image.resize((work_width,work_height),Image.Resampling.BILINEAR))
        started=time.monotonic()
        with torch.inference_mode():
            prediction=model.inference(image=[work_image],intrinsics=K_work[None].astype(np.float32),
                process_res=max(work_width,work_height),process_res_method='upper_bound_resize',export_dir=None)
        raw=np.asarray(prediction.depth,np.float32)
        if raw.shape!=(1,work_height,work_width):
            raise ValueError(f'DA3_UNEXPECTED_PIXEL_TRANSFORM: got {raw.shape}; never guess crop or K scaling')
        if prediction.processed_images is None or np.asarray(prediction.processed_images).shape[1:3]!=(work_height,work_width):
            raise ValueError('DA3_PROCESSED_IMAGE_GRID_MISMATCH')
        if model_name=='da3nested-giant-large' and requested_mode=='metric' and not bool(prediction.is_metric):
            raise ValueError('DA3_NESTED_METRIC_CLAIM_MISSING')
        flags.append(bool(prediction.is_metric))
        raw_path=output/'raw-work'/f'frame-{row["frame_index"]:06d}.npy'
        raw_path.parent.mkdir(parents=True,exist_ok=True);np.save(raw_path,raw[0])
        # Official Metric-Large outputs focal-normalized depth. Conversion uses
        # focal on the asserted model grid; Nested already did this internally.
        converted=raw[0]*np.float32(conversion)
        depth=np.asarray(Image.fromarray(converted).resize((width,height),Image.Resampling.BILINEAR),np.float32).copy()
        depth[~np.isfinite(depth)|(depth<=0)]=np.nan
        path=output/'depth'/f'frame-{row["frame_index"]:06d}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,depth)
        item={**artifact_ref(path),'frame_index':row['frame_index'],'timestamp_s':row['timestamp_s'],
            'valid':bool(np.isfinite(depth).any()),'raw_work_ref':artifact_ref(raw_path)}
        confidence=np.asarray(prediction.conf,np.float32) if prediction.conf is not None else None
        if confidence is not None:
            if confidence.shape!=raw.shape:raise ValueError('DA3_CONFIDENCE_SHAPE_MISMATCH')
            conf=np.asarray(Image.fromarray(confidence[0]).resize((width,height),Image.Resampling.BILINEAR),np.float32)
            cp=output/'depth'/f'confidence-{row["frame_index"]:06d}.npy';np.save(cp,conf);item['raw_model_confidence_ref']=artifact_ref(cp)
        finite=depth[np.isfinite(depth)]
        stats={'frame_index':row['frame_index'],'timestamp_s':row['timestamp_s'],
               'valid_fraction':float(finite.size/depth.size),'minimum':float(finite.min()) if finite.size else None,
               'percentiles_01_05_50_95_99':np.percentile(finite,[1,5,50,95,99]).tolist() if finite.size else None,
               'maximum':float(finite.max()) if finite.size else None,'inference_seconds':time.monotonic()-started}
        if previous is not None:
            valid=np.isfinite(previous)&np.isfinite(depth)
            change=np.abs(depth[valid]-previous[valid]);dt=float(row['timestamp_s'])-previous_time
            if dt<=0:raise ValueError('SOURCE_PTS_NOT_STRICTLY_INCREASING')
            stats['unwarped_temporal_abs_change_p50_p95']=np.percentile(change,[50,95]).tolist() if change.size else None
            stats['dt_s']=dt
        statistics.append(stats);previous=depth;previous_time=float(row['timestamp_s'])
        result_rows.append(item)
        write_json(checkpoint_path,{'run_signature':signature,'frame':item,'statistics':stats,'api_is_metric':flags[-1]})
        write_json(output/'progress.json',{'frames_done':ordinal+1,'frames_total':len(frames),'last_frame':stats,'training_ready':False})
        print(json.dumps({'frame':row['frame_index'],'frames_done':ordinal+1,'valid_fraction':stats['valid_fraction']},ensure_ascii=False),flush=True)
        if ordinal in {0,len(frames)//2,len(frames)-1}:
            _preview(image,depth,output/f'preview-{row["frame_index"]:06d}.png',requested_mode)
    try:code_revision=subprocess.check_output(['git','-C',str(native_path(request['model_repo'])),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):code_revision=None
    manifest={'schema_version':'learning.stage5.scene_depth.v1','source_key':request['source_key'],
              'source_video_sha256':source['source_video_sha256'],'mode':requested_mode,
              'quantity':'z_depth_m' if requested_mode=='metric' else 'relative_depth',
              'evidence_kind':'real_model','width':width,'height':height,
              'coordinate_frame':'C_optical','depth_axis':'z','camera':{'K':K_native.tolist(),'width':width,'height':height},
              'frames':result_rows,'provenance':{'model':'Depth Anything 3','model_dir':str(Path(request['model_dir']).resolve()),
              'model_name':model_name,'model_repo':persisted_path(native_path(request['model_repo'])),'code_revision':code_revision,
              'model_files':model_files,'official_code_files':code_files,'run_signature':signature,'worker_sha256':sha256(Path(__file__)),
              'api_is_metric_flags':flags,'scale_source':'official_checkpoint_prediction_and_dataset_K' if requested_mode=='metric' else 'relative_unscaled',
              'scale_is_sensor_ground_truth':False,'focal_work_px':focal_work,'raw_to_output_scale':conversion,
              'conversion_rule':'focal_work_px * net_output / 300' if model_name=='da3metric-large' and requested_mode=='metric' else 'identity',
              'work_size':[work_width,work_height],'K_work':K_work.tolist(),'pixel_affine_native_to_work':scale.tolist(),
              'pixel_transform_asserted':'model depth grid exactly equals explicit work grid; bilinear inverse resize to native',
              'native_resolution_inference':False,'inference_context':'single_view_per_frame','temporal_postprocessing':'none',
              'confidence_semantics':'raw model score; no calibrated [0,1] confidence is inferred',
              'code_api_ref':OFFICIAL_API,'prediction_spec_ref':OFFICIAL_SPEC,'official_metric_rule':OFFICIAL_METRIC,
              'torch_version':torch.__version__,'numpy_version':np.__version__,'python':sys.version,'interpreter':sys.executable,
              'gpu':torch.cuda.get_device_name() if device.startswith('cuda') else None,
              'peak_cuda_allocated_bytes':int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else 0,
              'compute_only':bool(request.get('compute_only',False)),'review_status':'pending','training_ready':False,'executed_at':utc_now()}}
    manifest['provenance']['model_dir']=persisted_path(native_path(request['model_dir']))
    provenance_ref=artifact_ref(write_json(output/'model-provenance.json',manifest['provenance']))
    manifest['measurement_provenance']=provenance_ref
    write_json(output/'statistics.json',{'frames':statistics,'units':'m' if requested_mode=='metric' else 'relative',
        'temporal_scope':'unwarped pixel comparison includes camera/object motion; descriptive, not GT error',
        'all_pixels_finite_positive':all(s['valid_fraction']==1. for s in statistics),'review_status':'pending'})
    write_json(final_path,manifest)
    return manifest


def _preview(image: Image.Image,depth: np.ndarray,output: Path,mode: str) -> None:
    """Viewable real RGB/depth pair; fixed metric color limits across frames."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,4.6),layout='constrained')
    axes[0].imshow(image);axes[0].set_title('Source RGB (unaltered)')
    vmin,vmax=(0,3) if mode=='metric' else tuple(np.nanpercentile(depth,[1,99]))
    plot=axes[1].imshow(depth,cmap='magma_r',vmin=vmin,vmax=vmax)
    axes[1].set_title('DA3 predicted Z depth; review pending')
    fig.colorbar(plot,ax=axes[1],label='camera Z (m)' if mode=='metric' else 'relative depth')
    for ax in axes:ax.axis('off')
    fig.savefig(output,dpi=130);plt.close(fig)


def attach_metric_anchors(depth_manifest: Path, anchors_manifest: Path, output_path: Path,
                          *, max_rmse_m: float, max_residual_m: float) -> dict:
    """Bind independent, hash-verified real measurements without editing raw depth.

    anchors_manifest: {source_key, frames:[{frame_index,anchors_ref:{path,sha256,bytes}}]}.
    Surface correspondence and actual fitting are validated by prepare_visual.
    """
    raw=read_json(depth_manifest);anchors=read_json(anchors_manifest)
    if raw.get('mode')!='relative' or raw['source_key']!=anchors.get('source_key'):
        raise ValueError('S5_ANCHOR_SOURCE_MISMATCH')
    if min(max_rmse_m,max_residual_m)<=0:raise ValueError('Residual limits must be explicit positive metric tolerances')
    mapping={int(row['frame_index']):row['anchors_ref'] for row in anchors['frames']}
    if len(mapping)!=len(anchors['frames']):raise ValueError('Duplicate anchor frame index')
    if set(mapping)!={r['frame_index'] for r in raw['frames']}:raise ValueError('Every depth frame requires a measured anchor set')
    for row in raw['frames']:
        item=mapping[row['frame_index']];verify_ref(item);data=read_json(item['path'])
        if data.get('source_key')!=raw['source_key'] or data.get('frame_index')!=row['frame_index']:
            raise ValueError('S5_ANCHOR_SOURCE_MISMATCH')
        if abs(float(data['timestamp_s'])-float(row['timestamp_s']))>1e-6:raise ValueError('S5_ANCHOR_TIMEBASE_MISMATCH')
        row['anchors_ref']=item
    raw['alignment']={'max_rmse_m':float(max_rmse_m),'max_residual_m':float(max_residual_m)}
    raw['anchor_binding_provenance']={'raw_manifest_ref':ref(depth_manifest),'anchors_manifest_ref':ref(anchors_manifest)}
    if output_path.resolve()==depth_manifest.resolve():raise ValueError('Raw depth manifest is immutable; provide a new output path')
    write_json(output_path,raw);return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--compute-only',action='store_true',help='Prepare source depth; never grant M5/M6/training approval')
    args=parser.parse_args();request=read_json(native_path(args.request))
    if args.compute_only:request['compute_only']=True
    args.output=native_path(args.output)
    state=preflight(request);args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'preflight.json',state)
    if args.execute:
        try:result=execute(request,args.output)
        except Exception as exc:
            write_json(args.output/'failure.json',{'type':type(exc).__name__,'error':str(exc),'no_mock_fallback':True});raise
        print(json.dumps({'frames':len(result['frames']),'mode':result['mode']}))
    else:print(json.dumps(state,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
