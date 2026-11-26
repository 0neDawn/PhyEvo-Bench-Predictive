"""Correct the boundary residual only orthogonally to observed state changes."""

import time
import torch


def align_orthogonal_boundary(prediction, history, *, rgb=False, component='orthogonal'):
    if component not in ('orthogonal','parallel'): raise ValueError(component)
    if (prediction.ndim < 3 or prediction.shape[1] < 2 or history.shape[1] < 2
        or prediction.shape[0] != history.shape[0] or prediction.shape[2:] != history.shape[2:]):
        raise ValueError('Matching history and at least two native forecast steps required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(history).all():
        raise ValueError('Non-finite states')
    value,observed=prediction.float(),history.float()
    residual=observed[:,-1:]-(2*value[:,:1]-value[:,1:2])
    # A has at most seven rows. Solve the small Gram problem, never a D x D matrix.
    changes=(observed[:,1:]-observed[:,:-1]).flatten(2).double()
    vector=residual.flatten(2).squeeze(1).double()
    gram=changes@changes.transpose(1,2)
    eigenvalues,eigenvectors=torch.linalg.eigh(gram)
    tolerance=(torch.finfo(torch.float64).eps*max(changes.shape[1:])
               *eigenvalues.amax(1,keepdim=True).clamp_min(0))
    keep=eigenvalues>tolerance
    inverse=torch.where(keep,1/eigenvalues.clamp_min(torch.finfo(torch.float64).tiny),0)
    rhs=(changes@vector.unsqueeze(-1))
    coefficients=eigenvectors@(inverse.unsqueeze(-1)*(eigenvectors.transpose(1,2)@rhs))
    dynamic=(changes.transpose(1,2)@coefficients).squeeze(-1)
    stable=(vector-dynamic).reshape_as(residual).float()
    correction=stable if component=='orthogonal' else dynamic.reshape_as(residual).float()
    unbounded=value+correction
    output=unbounded.clamp(0,1) if rgb else unbounded
    audit={
        'selected_component':component,
        'history_subspace_rank':keep.sum(1).tolist(),
        'boundary_residual_rms':float(residual.square().mean().sqrt()),
        'orthogonal_component_rms':float(stable.square().mean().sqrt()),
        'parallel_component_rms':float(dynamic.square().mean().sqrt()),
        'selected_correction_rms':float(correction.square().mean().sqrt()),
        'history_orthogonality_max_abs':float((changes@(vector-dynamic).unsqueeze(-1)).abs().max()),
        'prediction_change_rms':float((output-value).square().mean().sqrt()),
        'rgb_clipped_fraction':float(((unbounded<0)|(unbounded>1)).float().mean()) if rgb else 0.,
    }
    return output.to(prediction.dtype),audit


class OrthogonalBoundaryMixin:
    component='orthogonal'
    def _correct(self,prediction,history,*,rgb=False):
        if prediction.device.type=='cuda': torch.cuda.synchronize(prediction.device)
        started=time.perf_counter()
        with torch.inference_mode():
            output,self._projection_audit=align_orthogonal_boundary(prediction,history,rgb=rgb,component=self.component)
        if prediction.device.type=='cuda': torch.cuda.synchronize(prediction.device)
        self._projection_seconds=time.perf_counter()-started
        return output

    def score_pair(self,possible_rgb,impossible_rgb):
        result=super().score_pair(possible_rgb,impossible_rgb)
        self.last_track_audit['mode']=self.component+'_boundary'
        return result


class ParallelBoundaryMixin(OrthogonalBoundaryMixin):
    component='parallel'
