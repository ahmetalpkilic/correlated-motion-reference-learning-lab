"""Correlated reference positions -> GLS trajectory -> image-derived timestamp."""
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr
from scipy.optimize import least_squares

PIXELS = np.arange(-8., 9.)
SIGMA = .550 * 8 / (np.sqrt(2) * np.pi * 4)  # Gaussian pupil, in pixels
FLUX, BACKGROUND, READ_SD = 10000., 20., 2.
C, V, DELTA = .23, .8, .4  # pixels, pixels/second, seconds
REF_SD, COMMON_SD = .015, .012  # pixels


def image(center, sigma=SIGMA):
    """Integrate normalized Gaussian intensity over unit-width sensor pixels."""
    return BACKGROUND + FLUX * (ndtr((PIXELS+.5-center)/sigma)
                                - ndtr((PIXELS-.5-center)/sigma))


def center_derivative(center, sigma=SIGMA):
    zlo, zhi = (PIXELS-.5-center)/sigma, (PIXELS+.5-center)/sigma
    return FLUX / (sigma*np.sqrt(2*np.pi)) * (np.exp(-zlo*zlo/2)-np.exp(-zhi*zhi/2))


def fit_center(data):
    """Known PSF/photometry; one nonlinear parameter, independent Gaussian read noise."""
    signal = np.maximum(np.asarray(data)-BACKGROUND, 0)
    initial = np.clip(np.sum(PIXELS*signal)/np.sum(signal), -1.9, 1.9)
    r = least_squares(lambda x: (image(x[0])-data)/READ_SD, [initial],
                      jac=lambda x: center_derivative(x[0])[:, None]/READ_SD,
                      bounds=(-2., 2.), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not r.success:
        raise RuntimeError(r.message)
    return r.x[0]


def reference_covariance(n, common_sd=COMMON_SD):
    # One random shared offset per calibration run, plus independent position errors.
    return REF_SD**2*np.eye(n) + common_sd**2*np.ones((n, n))


def calibrate(times, positions, covariance):
    """Whiten before least squares; retain the origin/velocity cross covariance."""
    H = np.column_stack((np.ones(len(times)), times))
    L = np.linalg.cholesky(covariance)
    A, y = np.linalg.solve(L, H), np.linalg.solve(L, positions)
    if np.linalg.matrix_rank(A) < 2:
        raise ValueError('Reference times must identify both origin and velocity')
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    cov = np.linalg.solve(A.T@A, np.eye(2))
    return beta, cov


def timing(center, beta, center_variance, covariance):
    c, v = beta
    if abs(v) < 1e-8:
        raise ValueError('Stationary target cannot determine timing')
    delta = (center-c)/v
    g = np.array([-1/v, -delta/v])
    variance = center_variance/v**2 + g@covariance@g
    return delta, np.sqrt(variance)


def demo(out='outputs', trials=800):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261001)
    b = C+V*DELTA
    clean = image(b)
    bvar = READ_SD**2 / np.sum(center_derivative(b)**2)
    # Fit actual noisy pixel profiles in every trial; calibration is independent.
    centers = np.array([fit_center(clean+rng.normal(0, READ_SD, len(PIXELS)))
                        for _ in range(trials)])
    report = dict(seed=20261001, trials=trials, truth_delta_s=DELTA,
                  sigma_px=SIGMA, predicted_center_sd_px=np.sqrt(bvar),
                  empirical_center_sd_px=centers.std(ddof=1), scenarios={})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    axes[0].plot(PIXELS, clean-BACKGROUND, 'o-', label='pixel integrals')
    axes[0].set(xlabel='Sensor x (px)', ylabel='Signal (ADU)', title='Diffraction-limited spot at x=0.55 px', xlim=(-3, 3))
    for label, times in [('early', np.linspace(-3, -1, 5)),
                         ('bracketed', DELTA+np.linspace(-1, 1, 5))]:
        R = reference_covariance(len(times))
        nominal = C+V*times
        _, cov = calibrate(times, nominal, R)
        predictions = [timing(b, [C, V], bvar, cov)[1],
                       timing(b, [C, V], bvar, np.diag(np.diag(cov)))[1]]
        estimates, intervals, wrong_intervals = [], [], []
        for center in centers:
            obs = nominal + rng.normal(0, COMMON_SD) + rng.normal(0, REF_SD, len(times))
            beta, covariance = calibrate(times, obs, R)
            d, sd = timing(center, beta, bvar, covariance)
            estimates.append(d); intervals.append(sd)
            wrong_intervals.append(timing(center, beta, bvar, np.diag(np.diag(covariance)))[1])
        estimates = np.array(estimates)
        report['scenarios'][label] = dict(reference_times_s=times.tolist(),
            origin_velocity_covariance=cov.tolist(), correlation=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]),
            predicted_sd_s=predictions[0], diagonal_parameter_sd_s=predictions[1],
            empirical_sd_s=estimates.std(ddof=1), bias_s=float(estimates.mean()-DELTA),
            full_coverage_95=float(np.mean(abs(estimates-DELTA)<1.96*np.array(intervals))),
            diagonal_coverage_95=float(np.mean(abs(estimates-DELTA)<1.96*np.array(wrong_intervals))))
        axes[1].plot([0, 1, 2], 1000*np.array([*predictions, estimates.std(ddof=1)]), 'o-', label=label)
    mids = np.linspace(-3, 3, 121)
    design = []
    for midpoint in mids:
        t = midpoint+np.linspace(-1, 1, 5)
        _, cov = calibrate(t, C+V*t, reference_covariance(5))
        _, wrong = calibrate(t, C+V*t, np.diag(np.diag(reference_covariance(5))))
        design.append([timing(b, [C,V], bvar, cov)[1], timing(b, [C,V], bvar, wrong)[1]])
    design = np.array(design)
    report['design'] = dict(midpoints_s=mids.tolist(), timing_sd_s=design[:,0].tolist(),
        incorrectly_independent_reference_sd_s=design[:,1].tolist(), best_midpoint_s=float(mids[np.argmin(design[:,0])]),
        shared_offset_floor_s=COMMON_SD/abs(V))
    axes[1].set(xticks=[0,1,2], xticklabels=['Full covariance','Drop cross term','Monte Carlo'],
                ylabel='Timing SD (ms)', title='Covariance can help or hurt')
    axes[1].tick_params(axis='x', labelrotation=12)
    axes[2].plot(mids, 1000*design[:,0], label='correct shared error')
    axes[2].plot(mids, 1000*design[:,1], '--', label='wrong independent references')
    axes[2].axvline(DELTA, color='gray', linestyle=':', label='true capture time')
    axes[2].set(xlabel='Reference time midpoint (s)', ylabel='Predicted timing SD (ms)', title='Choose reference times near capture')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(out/'experiment.png', dpi=160); plt.close(fig)
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='design'}, indent=2))


if __name__ == '__main__':
    demo()
