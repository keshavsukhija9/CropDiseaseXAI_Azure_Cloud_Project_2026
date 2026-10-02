from dataclasses import dataclass, field
import numpy as np
from scipy.stats import ks_2samp


def population_stability_index(reference, current, bins=10):
    edges = np.histogram_bin_edges(reference, bins=bins)
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass
class DriftReport:
    input_psi: float
    class_dist_ks_stat: float
    class_dist_ks_pvalue: float
    confidence_psi: float
    drift_detected: bool
    reasons: list = field(default_factory=list)


PSI_ALERT_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05


def check_drift(reference_input_stat, current_input_stat, reference_class_ids,
                 current_class_ids, reference_confidence, current_confidence):
    input_psi = population_stability_index(reference_input_stat, current_input_stat)
    ks_stat, ks_p = ks_2samp(reference_class_ids, current_class_ids)
    conf_psi = population_stability_index(reference_confidence, current_confidence)

    reasons = []
    if input_psi > PSI_ALERT_THRESHOLD:
        reasons.append(f"input distribution PSI={input_psi:.3f} > {PSI_ALERT_THRESHOLD}")
    if ks_p < KS_PVALUE_THRESHOLD:
        reasons.append(f"class distribution KS p={ks_p:.4f} < {KS_PVALUE_THRESHOLD}")
    if conf_psi > PSI_ALERT_THRESHOLD:
        reasons.append(f"confidence PSI={conf_psi:.3f} > {PSI_ALERT_THRESHOLD}")

    return DriftReport(input_psi, ks_stat, ks_p, conf_psi, len(reasons) > 0, reasons)
