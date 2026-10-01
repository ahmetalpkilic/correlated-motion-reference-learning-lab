import unittest
import numpy as np
from scipy.integrate import quad
from lab import *

class PhysicsAndCalibration(unittest.TestCase):
    def test_pixel_integration_independent_quadrature(self):
        b = .55
        expected = [BACKGROUND+FLUX*quad(lambda x: np.exp(-.5*((x-b)/SIGMA)**2)/(SIGMA*np.sqrt(2*np.pi)), p-.5, p+.5, epsabs=1e-12)[0] for p in PIXELS]
        np.testing.assert_allclose(image(b), expected, atol=1e-10)
        self.assertAlmostEqual(np.sum(image(b)-BACKGROUND), FLUX, places=8)

    def test_gaussian_pupil_fourier_width(self):
        # P(r)=exp(-r²/(2a²)), a=f/(2N); sigma=lambda*f/(sqrt(8)*pi*a).
        f, N, wavelength, pitch = 16000., 8., .550, 4. # micrometres
        a=f/(2*N); r=np.linspace(-8*a, 8*a, 32768, endpoint=False)
        power=abs(np.fft.fftshift(np.fft.fft(np.exp(-.5*(r/a)**2))))**2
        x=np.fft.fftshift(np.fft.fftfreq(len(r), r[1]-r[0]))*wavelength*f/pitch
        # This amplitude convention yields sigma=lambda*N/(sqrt(2)*pi*pitch).
        width=np.sqrt(np.sum(power*x*x)/np.sum(power))
        self.assertAlmostEqual(width, SIGMA, places=10)

    def test_recovery_and_derivative(self):
        for b in [-.8,.55,1.2]:
            self.assertAlmostEqual(fit_center(image(b)), b, places=9)
            np.testing.assert_allclose(center_derivative(b), (image(b+1e-6)-image(b-1e-6))/2e-6, atol=1e-6)

    def test_closed_form_gls_covariance(self):
        t=np.linspace(-3,-1,5); m=t.mean(); S=np.sum((t-m)**2)
        beta,cov=calibrate(t, C+V*t, reference_covariance(5))
        expected=REF_SD**2*np.array([[1/5+m*m/S, -m/S],[-m/S,1/S]])
        expected[0,0]+=COMMON_SD**2
        np.testing.assert_allclose(beta,[C,V],atol=1e-14)
        np.testing.assert_allclose(cov,expected,atol=1e-14)

    def test_whitening_matches_dense_gls(self):
        t=np.array([-3.,-1.,0.,.4,2.]); H=np.c_[np.ones(5),t]
        R=reference_covariance(5); y=C+V*t+np.array([.01,-.03,.01,.02,.01])
        b,cov=calibrate(t,y,R)
        np.testing.assert_allclose(b,np.linalg.solve(H.T@np.linalg.solve(R,H), H.T@np.linalg.solve(R,y)),atol=1e-13)

    def test_design_optimum_and_common_floor(self):
        b=C+V*DELTA
        def variance(mid):
            t=mid+np.linspace(-1,1,5)
            _,cov=calibrate(t,C+V*t,reference_covariance(5))
            return timing(b,[C,V],0,cov)[1]**2
        best=variance(DELTA)
        self.assertAlmostEqual(best,(COMMON_SD**2+REF_SD**2/5)/V**2,places=14)
        self.assertGreater(variance(DELTA-1),best)
        self.assertGreater(variance(DELTA+1),best)

    def test_independent_monte_carlo_covariance(self):
        rng=np.random.default_rng(781)
        t=np.linspace(-3,-1,5); H=np.c_[np.ones(5),t]
        # Independent OLS form equals GLS for equal variances plus common offset.
        e=rng.normal(0,REF_SD,(30000,5))+rng.normal(0,COMMON_SD,(30000,1))
        samples=e@np.linalg.pinv(H).T
        _,cov=calibrate(t,C+V*t,reference_covariance(5))
        np.testing.assert_allclose(np.cov(samples.T),cov,rtol=.035)
        g=np.array([-1/V,-DELTA/V]); sd=np.sqrt(g@cov@g)
        errors=samples@g
        self.assertTrue(.94<np.mean(abs(errors)<1.96*sd)<.96)

    def test_degenerate_reference_times(self):
        with self.assertRaises(ValueError): calibrate(np.ones(5),np.ones(5),np.eye(5))
        with self.assertRaises(ValueError): timing(.2,[0,0],1,np.eye(2))

if __name__=='__main__': unittest.main()
