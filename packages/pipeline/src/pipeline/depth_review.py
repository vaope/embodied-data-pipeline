"""Create descriptive, viewable review evidence from real depth predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from pipeline.common import read_json, write_json
from pipeline.depth_worker import artifact_ref, checked_ref, native_path


def build_review(manifest_path: Path, source_index_path: Path, output: Path) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=read_json(native_path(manifest_path));source=read_json(native_path(source_index_path))
    if manifest['source_key']!=source['source_key']:raise ValueError('Source mismatch')
    if manifest.get('evidence_kind')!='real_model':raise ValueError('Review requires actual model predictions')
    output=native_path(output);output.mkdir(parents=True,exist_ok=True)
    rows=manifest['frames'];by_frame={int(r['frame_index']):r for r in source['frames']}
    timestamps=[];quantiles=[];changes=[];medians=[];fractions=[];previous=None
    conversion=float(manifest['provenance']['raw_to_output_scale'])
    direct_conversion_max_error=0.
    for row in rows:
        checked_ref(row);checked_ref(row['raw_work_ref'])
        rgb_row=by_frame[int(row['frame_index'])];checked_ref(rgb_row)
        if abs(float(rgb_row['timestamp_s'])-float(row['timestamp_s']))>1e-6:raise ValueError('PTS mismatch')
        depth=np.load(native_path(row['path']),allow_pickle=False)
        raw=np.load(native_path(row['raw_work_ref']['path']),allow_pickle=False)
        expected=np.asarray(Image.fromarray(raw*np.float32(conversion)).resize(
            (manifest['width'],manifest['height']),Image.Resampling.BILINEAR),np.float32)
        valid=np.isfinite(depth)&(depth>0)
        if valid.any():direct_conversion_max_error=max(direct_conversion_max_error,float(np.max(np.abs(depth[valid]-expected[valid]))))
        finite=depth[valid]
        timestamps.append(float(row['timestamp_s']));quantiles.append(np.percentile(finite,[1,5,50,95,99]).tolist())
        fractions.append(float(valid.mean()));medians.append(float(np.median(finite)))
        if previous is not None:
            common=valid&np.isfinite(previous)&(previous>0)
            changes.append(np.percentile(np.abs(depth[common]-previous[common]),[50,95]).tolist())
        previous=depth
    quantiles=np.asarray(quantiles);changes=np.asarray(changes)
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    axes[0].fill_between(timestamps,quantiles[:,0],quantiles[:,4],alpha=.15,label='pixel P01-P99')
    axes[0].fill_between(timestamps,quantiles[:,1],quantiles[:,3],alpha=.25,label='pixel P05-P95')
    axes[0].plot(timestamps,quantiles[:,2],label='median')
    axes[0].set_ylabel('Predicted camera Z (m)');axes[0].legend()
    axes[0].set_title('Real DA3Metric-Large predictions; learned scale, no depth ground truth')
    axes[1].plot(timestamps[1:],changes[:,0],label='P50 absolute change')
    axes[1].plot(timestamps[1:],changes[:,1],label='P95 absolute change')
    axes[1].set_ylabel('Unwarped frame difference (m)');axes[1].set_xlabel('Source video PTS (s)');axes[1].legend()
    axes[1].set_title('Includes camera/object motion; descriptive stability only')
    for ax in axes:ax.grid(alpha=.2)
    timeline=output/'depth-timeline.png';fig.savefig(timeline,dpi=140);plt.close(fig)
    selected=np.unique(np.linspace(0,len(rows)-1,min(6,len(rows)),dtype=int))
    color_max=float(max(1.,np.ceil(quantiles[:,4].max()*2)/2))
    fig,axes=plt.subplots(len(selected),2,figsize=(12,3.35*len(selected)),layout='constrained')
    for axes_row,index in zip(axes,selected):
        row=rows[index];rgb=Image.open(native_path(by_frame[int(row['frame_index'])]['path'])).convert('RGB')
        depth=np.load(native_path(row['path']),allow_pickle=False)
        axes_row[0].imshow(rgb);axes_row[0].set_title(f'Frame {row["frame_index"]}; PTS {row["timestamp_s"]:.3f}s')
        plot=axes_row[1].imshow(depth,cmap='magma_r',vmin=0,vmax=color_max)
        axes_row[1].set_title(f'Predicted Z; fixed 0-{color_max:g} m color scale');fig.colorbar(plot,ax=axes_row[1],label='m')
        for ax in axes_row:ax.axis('off')
    sheet=output/'depth-contact-sheet.png';fig.savefig(sheet,dpi=115);plt.close(fig)
    result={'schema_version':'learning.stage5.depth_review.v1','source_key':manifest['source_key'],
        'source_video_sha256':manifest['source_video_sha256'],'frames':len(rows),'source_pts_start_end_s':[timestamps[0],timestamps[-1]],
        'native_grid':[manifest['width'],manifest['height']],'model_processing_grid':manifest['provenance']['work_size'],
        'contact_sheet_color_scale_m':[0,color_max],
        'finite_positive_fraction_min_max':[min(fractions),max(fractions)],
        'frame_median_depth_min_max_m':[min(medians),max(medians)],
        'unwarped_temporal_abs_change_p50_m':np.percentile(changes[:,0],[50,95,100]).tolist(),
        'unwarped_temporal_abs_change_p95_m':np.percentile(changes[:,1],[50,95,100]).tolist(),
        'official_conversion_roundtrip_max_error_m':direct_conversion_max_error,
        'sensor_ground_truth_available':False,'accuracy_claim':'none; no measured scene depth reference',
        'review_status':'pending','training_ready':False,
        'artifacts':{'manifest':artifact_ref(native_path(manifest_path)),'source_index':artifact_ref(native_path(source_index_path)),
            'timeline':artifact_ref(timeline),'contact_sheet':artifact_ref(sheet)}}
    write_json(output/'depth-review.json',result);return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path);parser.add_argument('--source-index',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=build_review(args.manifest,args.source_index,args.output)
    print({k:v for k,v in result.items() if k!='artifacts'})


if __name__=='__main__':main()
