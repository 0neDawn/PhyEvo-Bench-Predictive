"""In-memory copies of native orchestration, with explicit insertion points.

No original source is edited. Each transformed function is saved for review.
Native neural modules, draw order, residuals and scoring remain unchanged.
"""
import hashlib
import inspect
import textwrap
import types

import torch

from .projection import historical_ball, project


class Hooks:
    def __init__(self, adapter, mode, snapshot_dir=None):
        if mode not in ("audit", "feedback", "output_only"):
            raise ValueError(mode)
        self.adapter, self.mode, self.snapshot_dir = adapter, mode, snapshot_dir
        self.restores, self.functions, self.audits = [], {}, []

    def replace(self, owner, name, function):
        original = getattr(owner, name)
        self.restores.append((owner, name, original))
        setattr(owner, name, function)

    def instrument(self, owner, name, replacements):
        original = getattr(owner, name)
        source = textwrap.dedent(inspect.getsource(original))
        transformed = source
        for before, after in replacements:
            if transformed.count(before) != 1:
                raise RuntimeError(f"Insertion point changed: {name}/{before!r}")
            transformed = transformed.replace(before, after, 1)
        namespace = dict(original.__func__.__globals__, _ppr=self)
        filename = f"<ppr_20261028/{owner.__class__.__name__}_{name}>"
        exec(compile(transformed, filename, "exec"), namespace)
        function = types.MethodType(namespace[name], owner)
        self.functions[name] = {"native_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                "instrumented_source_sha256": hashlib.sha256(transformed.encode()).hexdigest()}
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = self.snapshot_dir / f"{owner.__class__.__name__}_{name}.py"
            if path.exists():
                if path.read_text() != transformed:
                    raise RuntimeError("Instrumented source changed within a run")
            else:
                path.write_text(transformed)
        self.replace(owner, name, function)

    def correct(self, value):
        corrected, outside = project(value, self.center, self.radius, self.valid)
        if not torch.isfinite(corrected).all():
            raise RuntimeError("Nonfinite correction")
        self.current["steps"].append({"outside_objects": int(outside.sum()),
                                     "outside_by_object": outside.reshape(-1).cpu().tolist(),
                                     "change_l2_by_object": torch.linalg.vector_norm(corrected - value, dim=-1).reshape(-1).cpu().tolist(),
                                     "change_l2": float(torch.linalg.vector_norm(corrected - value))})
        return corrected

    def new_audit(self, counts):
        self.current = {"valid_objects": int(self.valid.sum()), "objects": int(self.valid.numel()),
                        "reliable_history_counts": counts.cpu().tolist(),
                        "radii": self.radius.cpu().tolist(), "steps": []}
        self.audits.append(self.current)

    def __exit__(self, *args):
        for owner, name, original in reversed(self.restores):
            setattr(owner, name, original)


class DDLPHooks(Hooks):
    def __enter__(self):
        model = self.adapter.model
        original = model.fg_sequential_opt
        def capture(*args, **kwargs):
            output = original(*args, **kwargs)
            self.presence = output["obj_on"].detach()
            self.mass = output["alpha_masks"].detach().flatten(2).sum(2)
            return output
        self.replace(model, "fg_sequential_opt", capture)
        original_decode = model.decode_all
        def capture_future(*args, **kwargs):
            output = original_decode(*args, **kwargs)
            self.current["future_visible_mass"] = output["alpha_masks"].detach().flatten(2).sum(2).cpu().tolist()
            self.current["future_position"] = args[0].detach().cpu().tolist()
            return output
        self.replace(model, "decode_all", capture_future)
        self.instrument(model.dyn_module, "sample", [
            ("bs, timestep_horizon, n_particles, _ = z.shape",
             "bs, timestep_horizon, n_particles, _ = z.shape\n    _ppr.begin(z_features)"),
            ("z = torch.cat([z, new_z], dim=1)",
             "if _ppr.mode == 'feedback':\n            new_z_features = _ppr.correct(new_z_features)\n        z = torch.cat([z, new_z], dim=1)"),
            ("return z, z_scale, z_obj_on, z_depth, z_features, z_bg_features",
             "if _ppr.mode == 'output_only':\n        changed = [_ppr.correct(z_features[:, t:t+1]) for t in range(timestep_horizon, z_features.shape[1])]\n        z_features = torch.cat([z_features[:, :timestep_horizon], *changed], dim=1)\n    return z, z_scale, z_obj_on, z_depth, z_features, z_bg_features")])
        return self

    def begin(self, features):
        if features.shape[0] != 1 or features.shape[1] != 8:
            raise RuntimeError("Expected frozen DDLP batch/history")
        history = features[0]
        self.center, self.radius, self.valid, counts = historical_ball(
            history, self.presence.reshape(history.shape[:2]), self.mass.reshape(history.shape[:2]))
        self.new_audit(counts)


class GSWMHooks(Hooks):
    def __enter__(self):
        fg = self.adapter.runtime.model.fg_module
        self.instrument(fg, "generate", [
            ("B, T, *_ = seq.size()", "B, T, *_ = seq.size()\n    _ppr.begin(B, cond_steps)"),
            ("fg, alpha_map = self.render(z)",
             "_ppr.before_render(t, ids, z)\n        fg, alpha_map = self.render(z)\n        _ppr.after_render(t, ids, z)")])
        self.instrument(fg, "render", [
            ("z_pres, z_depth, z_where, z_what, z_dyna = z",
             "z_pres, z_depth, z_where, z_what, z_dyna = z\n    if _ppr.mode == 'output_only' and _ppr.t >= _ppr.cond_steps:\n        z_what = _ppr.correct_by_id(z_what, _ppr.ids)"),
            ("return fg, alpha_map",
             "_ppr.mass = (importance_map * alpha_att_hat).sum(dim=(2, 3, 4)).detach()\n    return fg, alpha_map")])
        self.instrument(fg, "propagate_gen", [
            ("z = (z_pres, z_depth, z_where, z_what, z_dyna)",
             "if _ppr.mode == 'feedback':\n        z_what = _ppr.correct_by_id(z_what, _ppr.ids)\n    z = (z_pres, z_depth, z_where, z_what, z_dyna)")])
        return self

    def begin(self, batch, cond_steps):
        if batch != 1 or cond_steps != 8:
            raise RuntimeError("Expected frozen G-SWM batch/history")
        self.cond_steps, self.history, self.lookup = cond_steps, [], None
        self.t = -1

    def before_render(self, t, ids, z):
        self.t, self.ids = t, ids.detach().clone()

    def after_render(self, t, ids, z):
        if t >= self.cond_steps:
            self.current.setdefault("future_visible_mass", []).append(self.mass[0].cpu().tolist())
            self.current.setdefault("future_render_ids", []).append(ids[0].cpu().tolist())
            self.current.setdefault("future_where", []).append(z[2][0].detach().cpu().tolist())
            return
        self.history.append((ids[0].detach().clone(), z[3][0].detach().clone(),
                             z[0][0, :, 0].detach().clone(), self.mass[0].clone()))
        if t == self.cond_steps - 1:
            self.freeze_history()

    def freeze_history(self):
        # Native ids, never column order, determine temporal correspondence.
        unique = sorted({int(i) for ids, *_ in self.history for i in ids.tolist() if i >= 0})
        self.lookup = {identifier: col for col, identifier in enumerate(unique)}
        prototype = self.history[0][1]
        h = prototype.new_zeros((8, len(unique), prototype.shape[-1]))
        p = prototype.new_zeros((8, len(unique)))
        mass = torch.zeros_like(p)
        for t, (ids, features, pres, m) in enumerate(self.history):
            seen = set()
            for j, identifier in enumerate(ids.tolist()):
                if identifier < 0:
                    continue
                if identifier in seen:
                    raise RuntimeError("Duplicate native object id")
                seen.add(identifier)
                k = self.lookup[identifier]
                h[t, k], p[t, k], mass[t, k] = features[j], pres[j], m[j]
        self.center, self.radius, self.valid, counts = historical_ball(h, p, mass)
        self.new_audit(counts)
        self.current["object_ids"] = unique

    def correct_by_id(self, features, ids):
        if self.lookup is None:
            raise RuntimeError("Future correction before historical reference frozen")
        centers = torch.zeros_like(features)
        radii = features.new_zeros(features.shape[:-1])
        valid = torch.zeros(features.shape[:-1], dtype=torch.bool, device=features.device)
        for j, identifier in enumerate(ids[0].tolist()):
            if identifier in self.lookup:
                k = self.lookup[identifier]
                centers[0, j], radii[0, j], valid[0, j] = self.center[k], self.radius[k], self.valid[k]
        corrected, outside = project(features, centers, radii, valid)
        self.current["steps"].append({"outside_objects": int(outside.sum()),
                                     "object_ids": ids[0].cpu().tolist(),
                                     "outside_by_object": outside[0].cpu().tolist(),
                                     "change_l2_by_object": torch.linalg.vector_norm(corrected - features, dim=-1)[0].cpu().tolist(),
                                     "change_l2": float(torch.linalg.vector_norm(corrected - features))})
        if not torch.isfinite(corrected).all():
            raise RuntimeError("Nonfinite correction")
        return corrected
