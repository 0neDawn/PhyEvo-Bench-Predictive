"""Check native RNN repacking against observed module calls, not a guess."""
import inspect
import json
import os
import time

from . import common as c
from .parameter_runtime_003 import ALLOWED, content_hash, read_baseline, replace_once

GUARDS = {}
EXPECTED_STEPS = {1321: 33, 2103: 53}


def initialize_guard(adapter, model):
    if model != "slotformer":
        return
    from .initialization_audit_001 import parameter_state
    before = parameter_state(adapter)
    path = c.HERE / "weight_audits" / model / f"baseline_{os.getpid()}_{time.time_ns()}.json"
    payload = dict(at=time.time(), parameters=before, parameter_count=len(before),
                   value_manifest_sha256=content_hash(before),
                   runtime_addendum_sha256=c.digest(c.HERE / "PARAMETER_AUDIT_RUNTIME_ADDENDUM_004.json"))
    c.save(path, payload)
    GUARDS[id(adapter)] = dict(initial=before, previous=before, path=str(path), sha256=c.digest(path),
                              value_hash=payload["value_manifest_sha256"])


def audit_weights(adapter, model, baseline_versions, expected_trace, observed_lstm_calls):
    from .native_worker import versions
    if model != "slotformer":
        if versions(adapter, model) != baseline_versions:
            raise RuntimeError("Parameter identity/version changed")
        return dict(check="original_parameter_identity_and_version", all_versions_unchanged=True)
    from .initialization_audit_001 import parameter_state
    guard = GUARDS[id(adapter)]
    initial, previous = guard["initial"], guard["previous"]
    after = parameter_state(adapter)
    if after.keys() != initial.keys() or len(after) != 153:
        raise RuntimeError("SlotFormer parameter inventory changed")
    for name, value in after.items():
        for key in ("identity", "shape", "dtype", "requires_grad", "value_sha256"):
            if value[key] != initial[name][key]:
                raise RuntimeError(f"SlotFormer parameter changed: {name}/{key}")
    deltas = {name: value["version"] - previous[name]["version"] for name, value in after.items()
              if value["version"] != previous[name]["version"]}
    # clip_len=14 gives 7+13+13 LSTM calls. Starting at the original
    # clip_len=6 gives 7+7 history calls, 13+13 possible-target calls,
    # then13 impossible-target calls after clip_len has become14.
    expected_delta = EXPECTED_STEPS[expected_trace["module_forward_invocations"]]
    if observed_lstm_calls != expected_delta or set(deltas) != ALLOWED or set(deltas.values()) != {observed_lstm_calls}:
        raise RuntimeError(f"Native LSTM calls/version delta mismatch: calls={observed_lstm_calls},deltas={deltas}")
    observed_hash = content_hash(after)
    if observed_hash != guard["value_hash"]:
        raise RuntimeError("SlotFormer weight content changed")
    guard["previous"] = after
    return dict(check="all_parameter_identities_and_bitwise_values_native_lstm_version_audit",
                all_versions_unchanged=False, all_parameter_values_bitwise_unchanged=True,
                all_parameter_identities_unchanged=True, parameter_count=len(after),
                baseline_manifest=guard["path"], baseline_manifest_sha256=guard["sha256"],
                before_value_manifest_sha256=guard["value_hash"], after_value_manifest_sha256=observed_hash,
                version_deltas=deltas, expected_native_lstm_steps=expected_delta,
                observed_native_lstm_forward_calls=observed_lstm_calls,
                runtime_addendum_sha256=c.digest(c.HERE / "PARAMETER_AUDIT_RUNTIME_ADDENDUM_004.json"))


def verify_record(record):
    if record["model"] != "slotformer" or record.get("weight_identity_and_version_unchanged"):
        assert record["weight_identity_and_version_unchanged"]
        return
    audit = record["parameter_integrity_audit"]
    version3 = c.digest(c.HERE / "PARAMETER_AUDIT_RUNTIME_ADDENDUM_003.json")
    if audit["runtime_addendum_sha256"] == version3:
        # The three already completed no-op cases use the independently
        # validated1321/33 schedule. Their actual scores are retained.
        assert record["neural_call_trace"]["module_forward_invocations"] == 1321
        from .parameter_runtime_003 import verify_record as prior
        prior(record)
        return
    assert audit["check"] == "all_parameter_identities_and_bitwise_values_native_lstm_version_audit"
    assert audit["all_parameter_values_bitwise_unchanged"] and audit["all_parameter_identities_unchanged"]
    assert not audit["all_versions_unchanged"] and audit["parameter_count"] == 153
    assert record["weight_identity_and_values_unchanged"]
    expected = EXPECTED_STEPS[record["neural_call_trace"]["module_forward_invocations"]]
    assert set(audit["version_deltas"]) == ALLOWED and set(audit["version_deltas"].values()) == {expected}
    assert audit["expected_native_lstm_steps"] == audit["observed_native_lstm_forward_calls"] == expected
    manifest = read_baseline(audit["baseline_manifest"], audit["baseline_manifest_sha256"])
    assert audit["before_value_manifest_sha256"] == audit["after_value_manifest_sha256"] == manifest["value_manifest_sha256"]
    assert audit["runtime_addendum_sha256"] == manifest["runtime_addendum_sha256"] == c.digest(c.HERE / "PARAMETER_AUDIT_RUNTIME_ADDENDUM_004.json")


def worker_sources():
    from . import native_worker as worker
    run = inspect.getsource(worker.run)
    run = replace_once(run, "calls=[0];trace=hashlib.sha256()", "calls=[0];_lstm_calls=[0];trace=hashlib.sha256()")
    run = replace_once(run, "        def count(mod,inputs):\n            calls[0]+=1;",
                       "        def count(mod,inputs):\n"
                       "            if m=='slotformer' and mod is a.savi.predictor.rnn:_lstm_calls[0]+=1\n"
                       "            calls[0]+=1;")
    run = replace_once(run, "if versions(a,m)!=base:raise RuntimeError('Parameter identity/version changed')",
                       "_weight_audit = _audit_weights(a,m,base,expected,_lstm_calls[0])")
    run = replace_once(run, "weight_identity_and_version_unchanged=True,",
                       "weight_identity_and_version_unchanged=_weight_audit['all_versions_unchanged'],\n"
                       "            weight_identity_and_values_unchanged=_weight_audit.get('all_parameter_values_bitwise_unchanged'),\n"
                       "            parameter_integrity_audit=_weight_audit,")
    execute = replace_once(inspect.getsource(worker.execute), "base=versions(adapter,model);clip=",
                           "base=versions(adapter,model);_initialize_guard(adapter,model);clip=")
    return run, execute


def install_worker():
    from . import native_worker as worker
    run, execute = worker_sources()
    assert (c.HERE / "native_run_runtime_004.txt").read_text() == run
    assert (c.HERE / "native_execute_runtime_004.txt").read_text() == execute
    namespace = dict(vars(worker), _audit_weights=audit_weights, _initialize_guard=initialize_guard)
    exec(compile(run + "\n" + execute, "<O_parameter_integrity_runtime_004>", "exec"), namespace)
    worker.run, worker.execute = namespace["run"], namespace["execute"]


def install_report():
    from . import report
    source = replace_once(inspect.getsource(report.finish),
                          "assert rec['trace_matches_original'] and rec['extra_forwards']==0 and rec['weight_identity_and_version_unchanged']",
                          "assert rec['trace_matches_original'] and rec['extra_forwards']==0;_verify_weight_record(rec)")
    assert (c.HERE / "report_finish_runtime_004.txt").read_text() == source
    namespace = dict(vars(report), _verify_weight_record=verify_record)
    exec(compile(source, "<O_final_weight_integrity_runtime_004>", "exec"), namespace)
    report.finish = namespace["finish"]


def verify_addendum():
    from .parameter_runtime_003 import verify_addendum as prior
    prior()
    for path, sha256 in c.read(c.HERE / "PARAMETER_AUDIT_RUNTIME_ADDENDUM_004.json")["files"].items():
        if c.digest(path) != sha256:
            raise RuntimeError("Frozen runtime source changed: " + path)
