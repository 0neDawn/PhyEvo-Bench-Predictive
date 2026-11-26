"""Only prediction and history enter the correction; no candidate arguments."""
import torch
from predictive_tracks.ppr_all_20261029.projection import reference,step

def trajectory(prediction,history,alpha):
    if alpha<1:raise ValueError('alpha must retain all history')
    if prediction.ndim!=4 or prediction.shape[0]!=1 or prediction.shape[2:]!=history.shape[2:]:
        raise ValueError('Original tensor layout required')
    c,r,valid=reference(history);scaled=alpha*r
    values=[];audits=[]
    for t in range(prediction.shape[1]):
        y,audit=step(prediction[:,t],(c,scaled,valid));values.append(y);audits.append(audit)
    return torch.stack(values,dim=1),{'alpha':alpha,'eta':1.,'history_steps':history.shape[1],
        'radii_min_mean_max':[float(scaled.min()),float(scaled.mean()),float(scaled.max())],
        'base_radii_min_mean_max':[float(r.min()),float(r.mean()),float(r.max())],'steps':audits}

def pooled_trajectory(prediction,history,alpha):
    pp=prediction.mean(2,keepdim=True);hh=history.mean(2,keepdim=True)
    fixed,audit=trajectory(pp,hh,alpha)
    out=prediction+(fixed-pp)
    if not torch.isfinite(out).all():raise RuntimeError('Nonfinite correction')
    audit.update(interface='spatial_mean_same_offset_all_tokens',spatial_units=prediction.shape[2],
        field_change_l2=float(torch.linalg.vector_norm(out-prediction)),
        spatial_contrast_max_abs_roundoff=float(((out-out[:,:,:1])-(prediction-prediction[:,:,:1])).abs().max()))
    return out,audit
