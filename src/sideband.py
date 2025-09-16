from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import numpy as np
from signal_processing import list_signals, TS


def is_local_peak(amps, idx, window=5):
    """
    Checks if a given index in an amplitude array is a local peak within a specified window.

    Parameters:
    - amps (np.ndarray): 1D array of amplitudes
    - idx (int): index to check
    - window (int, optional): number of neighboring points on each side to consider (default: 5)

    Returns:
    - bool: True if amps[idx] is a local peak, False otherwise
    """
    left = max(0, idx - window)
    right = min(len(amps), idx + window + 1)
    return all(amps[idx] >= amps[i] for i in range(left, right) if i != idx)


def detect_final_gmfs3(ts, prominence_ratio=0.01, tolerance=1.5, min_support=3, max_initial_peaks=5):
    """
    Detects Generalized Mechanical Frequencies (GMFs) in a single signal.

    The function finds peaks in the FFT spectrum, filters them by local peak condition,
    minimum frequency, prominence, and harmonic support, then selects the final GMFs.

    Parameters:
    - ts: signal object with attributes fq (frequencies) and ft (FFT amplitudes)
    - prominence_ratio (float, optional): relative prominence threshold for peaks (default: 0.01)
    - tolerance (float, optional): frequency tolerance for harmonic matching (default: 1.5)
    - min_support (int, optional): minimum number of harmonic matches to accept a GMF (default: 3)
    - max_initial_peaks (int, optional): number of top peaks to consider as candidates (default: 5)

    Returns:
    - list of float: detected GMFs
    """
    if not hasattr(ts, 'fq') or not hasattr(ts, 'ft') or ts.fq is None or ts.ft is None or len(ts.ft) == 0:
        ts.fftransform()

    freqs = ts.fq
    amps = ts.ft

    max_amp = np.max(amps)
    prom_thresh = max_amp * prominence_ratio
    peak_indices, props = find_peaks(amps, prominence=prom_thresh, distance=3)
    peak_indices = [idx for idx in peak_indices if is_local_peak(amps, idx)]
    peak_indices = np.array(peak_indices)

    peak_freqs = freqs[peak_indices]
    peak_amps = amps[peak_indices]

    mask = peak_freqs >= 20
    peak_freqs = peak_freqs[mask]
    peak_amps = peak_amps[mask]
    peak_indices = peak_indices[mask]

    sorted_peaks = sorted(zip(peak_freqs, peak_amps),
                          key=lambda x: x[1], reverse=True)
    current_peak_freqs = [f for f, _ in sorted_peaks]

    final_gmfs = []
    used_freqs = set()

    for base_idx, (candidate_gmf, _) in enumerate(sorted_peaks[:max_initial_peaks]):
        if candidate_gmf in used_freqs:
            continue

        harmonics_found = 0
        harmonics_freqs = []

        for i in range(1, 10):
            harmonic_freq = candidate_gmf * i
            close_peaks = [f for f in current_peak_freqs if abs(
                f - harmonic_freq) <= tolerance]
            if close_peaks:
                harmonics_found += 1
                harmonics_freqs.extend(close_peaks)

        if harmonics_found >= min_support:
            final_gmfs.append((candidate_gmf, harmonics_found))
            used_freqs.update(harmonics_freqs)

            current_peak_freqs = [f for f in current_peak_freqs if all(
                abs(f - hf) > tolerance for hf in harmonics_freqs)]
    return [gmf for gmf, _ in final_gmfs]


def detect_final_gmfs_across_signals_ref(
    ts_list,
    ref_idx=0,
    prominence_ratio=0.01,
    tolerance=2.5,
    max_initial_peaks=5,
    min_freq=0,
    min_harmonics=2,
    max_harmonic=5,
    verbose=True
):
    """
    Detects GMFs consistently across multiple signals using a reference signal.

    Each signal is processed to find peaks. Candidates are filtered by harmonic presence
    across signals, amplitude thresholds, and multiplicity checks to obtain final GMFs.

    Parameters:
    - ts_list (list): list of signal objects
    - ref_idx (int, optional): index of reference signal in ts_list (default: 0)
    - prominence_ratio (float, optional): relative prominence threshold for peaks (default: 0.01)
    - tolerance (float, optional): frequency tolerance for harmonics (default: 2.5)
    - max_initial_peaks (int, optional): maximum peaks considered per signal (default: 5)
    - min_freq (float, optional): minimum frequency to consider (default: 0)
    - min_harmonics (int, optional): minimum harmonic count to accept a GMF (default: 2)
    - max_harmonic (int, optional): maximum harmonic order to check (default: 5)
    - verbose (bool, optional): whether to print intermediate information (default: True)

    Returns:
    - list of float: final GMFs found across all signals
    """

    if len(ts_list) == 0:
        return []

    all_peak_freqs = []
    all_peak_amps = []
    for ts in ts_list:
        if not hasattr(ts, 'fq') or not hasattr(ts, 'ft') or ts.fq is None or ts.ft is None or len(ts.ft) == 0:
            ts.fftransform()

        freqs = ts.fq
        amps = ts.ft
        if len(amps) == 0:
            all_peak_freqs.append(np.array([]))
            all_peak_amps.append(np.array([]))
            continue

        prom_thresh = np.max(amps) * prominence_ratio
        peak_indices, props = find_peaks(
            amps, prominence=prom_thresh, distance=3)
        peak_freqs = freqs[peak_indices]
        peak_amps = amps[peak_indices]

        mask = peak_freqs >= min_freq
        peak_freqs = peak_freqs[mask]
        peak_amps = peak_amps[mask]
        gmfs = detect_final_gmfs3(ts)
        all_peak_freqs.append(np.array(gmfs))
        all_peak_amps.append(np.array(peak_amps))

    ref_peak_freqs = all_peak_freqs[ref_idx]
    ref_peak_amps = all_peak_amps[ref_idx]
    if len(ref_peak_freqs) == 0:
        if verbose:
            print("Reference signal has no peaks >= min_freq.")
        return []

    sorted_ref = sorted(zip(ref_peak_freqs, ref_peak_amps),
                        key=lambda x: x[1], reverse=True)
    candidates = [f for f, a in sorted_ref[:max_initial_peaks]]
    final_gmfs = []

    filtered_candidates = []
    candidates_sorted = sorted(candidates)

    for i, g1 in enumerate(candidates_sorted):
        keep = True
        for j, g2 in enumerate(candidates_sorted):
            if i != j and g1 % g2 == 0:
                if g1 > g2:
                    keep = False
                    break
        if keep:
            filtered_candidates.append(g1)

    candidates = filtered_candidates

    for gmf in candidates:
        gmf_amps = []
        peak_count = 0

        for ts in ts_list:
            if not hasattr(ts, 'fq') or not hasattr(ts, 'ft') or ts.fq is None or ts.ft is None:
                ts.fftransform()
            freqs = ts.fq
            amps = ts.ft

            mask = (freqs >= gmf - 1) & (freqs <= gmf + 1)
            local_indices, _ = find_peaks(
                amps[mask], prominence=np.max(amps[mask])*0.1)

            if len(local_indices) > 0:
                peak_count += 1
                local_max = np.max(amps[mask])
                gmf_amps.append(local_max)

        if peak_count / len(ts_list) < 0.8:
            continue

        harmonic_count = 0
        for h in range(2, max_harmonic + 1):
            harmonic_freq = gmf * h
            found = False
            for ts in ts_list:
                freqs = ts.fq
                mask = (freqs >= harmonic_freq - tolerance) & (freqs <=
                                                               harmonic_freq + tolerance)
                if np.any(mask):
                    found = True
                    break
            if found:
                harmonic_count += 1

        if gmf < 30:
            if gmf_amps[0] < 0.6 * max(amps):
                continue

        if harmonic_count >= min_harmonics:
            final_gmfs.append(gmf)
            if verbose:
                total_amp = sum(gmf_amps)

    cleaned_gmfs = []
    for g in sorted(final_gmfs):
        if not any(abs(g - n * base) <= tolerance for base in cleaned_gmfs for n in range(2, 10)):
            cleaned_gmfs.append(g)

    final_gmfs = cleaned_gmfs

    if verbose:
        if len(final_gmfs) == 0:
            print("No GMF found across all signals with the given min_support.")

    return final_gmfs


def calculate_score_iso(value, iso=[1, 2]):
    lower_bound = 0.8*iso[0]
    upper_bound = 1.2*iso[1]

    score = (value - lower_bound) / (upper_bound - lower_bound)
    score = np.clip(score, 0, 1)
    score = np.round(score, 2)
    return score


def evaluate_sideband_set(
    harm_amp,
    sideband_amps,
    expected_count=None,
    *,
    w_cnt=0.8,
    w_amp=0.2,
    min_amp_ratio=0.01,
    max_amp_ratio=0.6,
    min_sidebands=3
):
    """
    Evaluates the quality of a set of sidebands relative to a harmonic amplitude.

    Combines count-based and amplitude-based scoring to produce a final weighted score.

    Parameters:
    - harm_amp (float): amplitude of the main harmonic
    - sideband_amps (list of float): amplitudes of sideband peaks
    - expected_count (int, optional): expected number of sidebands (default: max of valid counts or min_sidebands)
    - w_cnt (float, optional): weight for count score (default: 0.8)
    - w_amp (float, optional): weight for amplitude score (default: 0.2)
    - min_amp_ratio (float, optional): minimum sideband amplitude ratio relative to harmonic (default: 0.01)
    - max_amp_ratio (float, optional): maximum sideband amplitude ratio before penalty (default: 0.6)
    - min_sidebands (int, optional): minimum number of sidebands (default: 3)

    Returns:
    - dict: {"score": final_score, "amp_score": amplitude_component, "count_score": count_component}
    """
    if harm_amp is None or harm_amp <= 0 or not sideband_amps:
        return {
            "score": 0.0,
            "amp_score": 0.0,
            "count_score": 0.0
        }

    ratios = [a / harm_amp for a in sideband_amps if a /
              harm_amp >= min_amp_ratio]
    n_valid = len(ratios)

    if expected_count is None:
        expected_count = max(n_valid, min_sidebands)

    count_score = calculate_score_iso(
        n_valid, iso=[min_sidebands, 30])

    if ratios:
        mean_ratio = np.mean(ratios)
        spread = (np.max(ratios) - np.min(ratios)) if len(ratios) > 1 else 0.0
        amp_score = mean_ratio * (1.0 - 0.5 * spread)
    else:
        amp_score = 0.0

    if n_valid < 5:
        penalty = sum(max(r - max_amp_ratio, 0.0) for r in ratios)
        amp_score = max(amp_score - penalty, 0.0)

    score = 100.0 * (w_cnt * count_score + w_amp * amp_score)

    return {
        "score": round(score, 2),
        "amp_score": round(amp_score, 4),
        "count_score": round(count_score, 4)
    }


def find_sidebands_harmonics_filtered(ts, gmfs, min_prom_ratio=0.03,
                                      min_train_len=3, smoothing_sigma=1.0,
                                      base_peak_min_ratio=0.1, min_spacing=2,
                                      tol_hz=1.0):
    """
    Detects sidebands for given GMFs in a signal using harmonic alignment and amplitude thresholds.

    Parameters:
    - ts: signal object with fq (frequencies) and ft (FFT amplitudes)
    - gmfs (list of float): GMF candidates to check for sidebands
    - min_prom_ratio (float): minimum prominence ratio for peak detection
    - min_train_len (int): minimum consecutive sidebands to consider valid
    - smoothing_sigma (float): sigma for Gaussian smoothing of FFT
    - base_peak_min_ratio (float): minimum ratio of base peak amplitude relative to max amplitude
    - min_spacing (float): minimum spacing between consecutive sidebands
    - tol_hz (float): frequency tolerance for excluding harmonics

    Returns:
    - list of dict: each dict contains {"gmf": value, "spacing": estimated_spacing, "score": sideband_score}
    """

    if not hasattr(ts, 'fq') or ts.fq is None or ts.ft is None:
        ts.fftransform()
    freqs = ts.fq
    amps = ts.ft
    amps_smooth = gaussian_filter1d(amps, sigma=smoothing_sigma)

    results = []

    global_prom_thresh = np.max(amps_smooth) * min_prom_ratio
    peak_indices, _ = find_peaks(
        amps_smooth, prominence=global_prom_thresh, distance=1)
    all_peak_freqs = freqs[peak_indices]
    all_peak_amps = amps_smooth[peak_indices]

    reserved_freqs = []

    def exclusion_mask(target_gmf):
        mask = np.ones_like(all_peak_freqs, dtype=bool)

        for g in gmfs:
            if np.isclose(g, target_gmf):
                continue
            H = int(np.floor(freqs.max() / g))
            for h in range(1, H + 1):
                c = h * g
                mask &= ~((all_peak_freqs >= c - tol_hz) &
                          (all_peak_freqs <= c + tol_hz))

        for rf in reserved_freqs:
            mask &= ~((all_peak_freqs >= rf - tol_hz) &
                      (all_peak_freqs <= rf + tol_hz))

        return mask

    for gmf in gmfs:
        mask = exclusion_mask(gmf)
        peak_freqs = all_peak_freqs[mask]
        peak_amps = all_peak_amps[mask]

        sidebands_by_harmonic = {1: [], 2: [], 3: []}
        harmonic_scores = []

        base_peak_candidates_idx = np.where(
            np.abs(peak_freqs - gmf) <= gmf*0.1)[0]
        if len(base_peak_candidates_idx) == 0:
            results.append({"gmf": gmf, "spacing": None, "score": 0.0})
            continue

        idx_base = base_peak_candidates_idx[np.argmax(
            peak_amps[base_peak_candidates_idx])]
        base_amp = peak_amps[idx_base]

        if base_amp < np.max(amps_smooth) * base_peak_min_ratio:
            results.append({"gmf": gmf, "spacing": None, "score": 0.0})
            continue

        prev_harmonic_has_sb = True

        for h in [1, 2, 3]:
            if not prev_harmonic_has_sb:
                break

            center = h * gmf
            idx_h_peak = np.where(np.abs(peak_freqs - center) <= gmf*0.01)[0]
            if len(idx_h_peak) == 0:
                prev_harmonic_has_sb = False
                continue

            idx_h_best = idx_h_peak[np.argmax(peak_amps[idx_h_peak])]
            harmonic_amp = peak_amps[idx_h_best]

            if harmonic_amp < np.max(amps_smooth) * 0.3:
                prev_harmonic_has_sb = False
                continue

            idx_h_candidates = np.where(
                np.abs(peak_freqs - center) <= gmf*0.6)[0]
            if len(idx_h_candidates) == 0:
                prev_harmonic_has_sb = False
                continue

            amps_h = peak_amps[idx_h_candidates]
            freqs_h = peak_freqs[idx_h_candidates]

            aligned_h = []
            sorted_idx = np.argsort(freqs_h)
            last_freq = None
            for i in sorted_idx:
                if last_freq is not None and abs(freqs_h[i] - last_freq) < min_spacing:
                    if amps_h[i] > amps_h[i-1]:
                        aligned_h[-1] = freqs_h[i]
                        last_freq = freqs_h[i]
                    continue
                aligned_h.append(freqs_h[i])
                last_freq = freqs_h[i]

            if len(aligned_h) >= min_train_len:
                sidebands_by_harmonic[h].extend(aligned_h)
                prev_harmonic_has_sb = True

                eval_out = evaluate_sideband_set(
                    harmonic_amp,
                    [peak_amps[np.argmin(np.abs(peak_freqs - sb))]
                     for sb in aligned_h],
                    expected_count=max(len(aligned_h), 3)
                )
                harmonic_scores.append(eval_out['score'])
            else:
                prev_harmonic_has_sb = False

        sidebands_all = []
        for h in sidebands_by_harmonic:
            sidebands_all.extend(sidebands_by_harmonic[h])
        sidebands_all_sorted = np.sort(sidebands_all)

        spacing_est = np.median(np.diff(sidebands_all_sorted)) if len(
            sidebands_all_sorted) > 1 else gmf
        total_score = np.mean(harmonic_scores) if harmonic_scores else 0.0

        results.append({
            "gmf": gmf,
            "spacing": spacing_est,
            "score": round(total_score, 2)
        })

        reserved_freqs.append(gmf)
        reserved_freqs.extend([2*gmf, 3*gmf])
        reserved_freqs.extend(sidebands_all_sorted.tolist())

    return results


def aggregate_from_results(list_of_results, tol=0.5, gmf_tol_ratio=0.1):
    """
    Aggregates sideband spacing results from multiple runs/signals.

    Clusters similar spacing values for each GMF and returns the average spacing of the largest cluster.

    Parameters:
    - list_of_results (list of list of dict): results from multiple sideband searches
    - tol (float): tolerance to consider spacings part of the same cluster
    - gmf_tol_ratio (float): fraction of GMF frequency to ignore spacings too close to the GMF

    Returns:
    - dict: {gmf_value: final_spacing}
    """
    gmfs_spacings = defaultdict(list)

    for results in list_of_results:
        for r in results:
            if r["spacing"] is not None:
                gmfs_spacings[r["gmf"]].append(r["spacing"])

    final_spacings = {}
    for gmf, spacings in gmfs_spacings.items():
        clean_spacings = [
            s for s in spacings if abs(s - gmf) > gmf * gmf_tol_ratio
        ]

        if len(clean_spacings) == 0:
            final_spacings[gmf] = 0.0
            continue

        spacings = np.array(clean_spacings)
        used = np.zeros(len(spacings), dtype=bool)
        clusters = []

        for i, s in enumerate(spacings):
            if used[i]:
                continue
            cluster = [s]
            used[i] = True
            for j in range(i+1, len(spacings)):
                if not used[j] and abs(spacings[j] - s) <= tol:
                    cluster.append(spacings[j])
                    used[j] = True
            clusters.append(cluster)

        best_cluster = max(clusters, key=len)
        final_spacings[gmf] = np.mean(best_cluster)

    return final_spacings


def evaluate_sidebands_with_final_spacing_stepwise(
    ts, gmfs_final, spacing_map,
    tol_hz=0.5,
    min_prom_ratio=0.03,
    min_train_len=2,
    smoothing_sigma=1.0,
    base_peak_min_ratio=0.05,
    allowed_gaps=2,
    max_steps_limit=50,
):
    """
    Performs stepwise search for sidebands around each harmonic using a fixed spacing.

    Searches in both directions from each harmonic, allows a limited number of missing steps,
    and scores sidebands based on amplitude relative to the harmonic.

    Parameters:
    - ts: signal object with fq (frequencies) and ft (FFT amplitudes)
    - gmfs_final (list of float): final GMFs to process
    - spacing_map (dict): {gmf: spacing} map for stepwise search
    - tol_hz (float): frequency tolerance to locate peaks
    - min_prom_ratio (float): minimum peak prominence ratio
    - min_train_len (int): minimum number of consecutive sidebands to accept
    - smoothing_sigma (float): sigma for Gaussian smoothing
    - base_peak_min_ratio (float): minimum amplitude ratio of harmonic to max amplitude
    - allowed_gaps (int): number of missing steps allowed before stopping in a direction
    - max_steps_limit (int): maximum steps to search in each direction

    Returns:
    - list of dict: each dict contains {"gmf": value, "sidebands": {harmonic_idx: [sideband_indices]}, "spacing": spacing}
    """

    if not hasattr(ts, 'fq') or ts.fq is None or ts.ft is None:
        ts.fftransform()
    freqs = ts.fq
    amps = ts.ft
    amps_smooth = gaussian_filter1d(amps, sigma=smoothing_sigma)

    global_prom_thresh = np.max(amps_smooth) * min_prom_ratio
    peak_indices, _ = find_peaks(amps_smooth, prominence=global_prom_thresh)
    peak_freqs = freqs[peak_indices]
    peak_amps = amps_smooth[peak_indices]

    results = []

    for gmf in gmfs_final:
        spacing = spacing_map.get(gmf, 0)
        if spacing == 0:
            results.append({"gmf": gmf, "sidebands": {},
                           "spacing": 0.0})
            continue

        tol = tol_hz if tol_hz is not None else max(0.5, 0.02 * gmf)

        sidebands_dict = {}
        harmonic_scores = []

        for h in [1, 2, 3]:
            center = h * gmf

            idx_h = np.where(np.abs(peak_freqs - center) <= tol)[0]
            if len(idx_h) == 0:
                continue

            best_local = np.argmax(peak_amps[idx_h])
            harmonic_idx = peak_indices[idx_h[best_local]]
            harmonic_amp = peak_amps[idx_h[best_local]]

            if harmonic_amp < np.max(amps_smooth) * base_peak_min_ratio:
                continue

            found_sbs_idx = []

            def find_peak_near(freq_target):
                idx = np.where(np.abs(peak_freqs - freq_target) <= tol)[0]
                if len(idx) == 0:
                    return None
                best_local = np.argmax(peak_amps[idx])
                best_global = idx[best_local]
                return peak_freqs[best_global], peak_amps[best_global], peak_indices[best_global]

            gaps = 0
            steps = 0
            n = 1
            while steps < max_steps_limit:
                freq_target = center + n * spacing
                if freq_target > freqs.max():
                    break
                res = find_peak_near(freq_target)
                if res is not None:
                    found_sbs_idx.append(res[2])
                    gaps = 0
                else:
                    gaps += 1
                    if gaps > allowed_gaps:
                        break
                n += 1
                steps += 1

            gaps = 0
            steps = 0
            n = 1
            while steps < max_steps_limit:
                freq_target = center - n * spacing
                if freq_target < freqs.min():
                    break
                res = find_peak_near(freq_target)
                if res is not None:
                    found_sbs_idx.append(res[2])
                    gaps = 0
                else:
                    gaps += 1
                    if gaps > allowed_gaps:
                        break
                n += 1
                steps += 1

            if len(found_sbs_idx) >= min_train_len:
                sidebands_dict[harmonic_idx] = sorted(set(found_sbs_idx))

                amplitudes = [amps_smooth[i] for i in found_sbs_idx]
                score = np.mean([amp / harmonic_amp for amp in amplitudes])
                harmonic_scores.append(score)
            else:
                sidebands_dict[harmonic_idx] = []

        results.append({
            "gmf": gmf,
            "sidebands": sidebands_dict,
            "spacing": spacing,
        })

    return results


def find_signals_sideband(path):
    signals = list_signals(path)

    sigs = []
    for signal in signals:
        if 'Vel' not in signal:
            continue

        sig = TS(signal)
        if sig.details.iloc[3][0][-1] == 'g':
            sig.signal /= 9.806
        sig.fftransform()
        sigs.append(sig)

    final_gmfs = detect_final_gmfs_across_signals_ref(sigs)

    all_results = []
    all_scores = []
    for s in sigs:
        sideband = find_sidebands_harmonics_filtered(s, final_gmfs)
        all_results.append(sideband)

        scores_per_gmf = {r['gmf']: r['score'] for r in sideband}
        all_scores.append(scores_per_gmf)

    final_spacings = aggregate_from_results(all_results, tol=0.5)

    final_dict = {}

    for sig_idx, s in enumerate(sigs):
        refined = evaluate_sidebands_with_final_spacing_stepwise(
            s, final_gmfs, final_spacings
        )

        scores_for_sig = all_scores[sig_idx]

        sig_name = s.name
        final_dict[sig_name] = []

        for gmf_entry in refined:
            gmf_value = gmf_entry['gmf']
            gmf_dict = {
                'gmf': gmf_value,
                'score': scores_for_sig.get(gmf_value, None),
                'sidebands': gmf_entry.get('sidebands', {}),
                'spacing': gmf_entry.get('spacing', None)
            }
            final_dict[sig_name].append(gmf_dict)

    return final_dict
