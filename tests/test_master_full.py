"""
MASTER FULL TEST SUITE — havok-toolbox
=======================================
Unit | Integration | Regression | Stress | Edge Cases | Contract | CLI | Serialization
"""
import numpy as np, pytest, warnings, os, io, tempfile, time

@pytest.fixture(scope="module")
def clean_sine(): t=np.linspace(0,50,3000); return np.sin(2*np.pi*0.1*t)
@pytest.fixture(scope="module")
def noisy_sine(): rng=np.random.default_rng(42); t=np.linspace(0,50,3000); return np.sin(2*np.pi*0.1*t)+rng.normal(0,0.5,3000)
@pytest.fixture(scope="module")
def lorenz_x():
    rng=np.random.default_rng(0); x=np.zeros(4000); x[0]=0.1
    for i in range(1,4000): x[i]=x[i-1]+0.01*(10*(x[i-1]+rng.normal(0,0.01))-x[i-1]); x[i]=np.clip(x[i],-50,50)
    return x
@pytest.fixture(scope="module")
def regime_shift_signal():
    rng=np.random.default_rng(7)
    p1=np.sin(np.linspace(0,20,2000))+rng.normal(0,0.05,2000)
    p2=np.sin(np.linspace(0,20,2000))*3+rng.normal(0,0.5,2000)
    return np.concatenate([p1,p2])
@pytest.fixture(scope="module")
def fitted_estimator(clean_sine):
    from havolib.estimator import HavokEstimator
    return HavokEstimator(r=5,tau=2,m=20,diff_method="finite_diff").fit(clean_sine)

class TestImports:
    def test_all_modules_import(self):
        for m in ["estimator","pipeline","adaptive","arena","attribution","auto_tune","automl","config",
                  "data_loader","decomposition","detection","edge_of_chaos","embedding","federated",
                  "forcing","gpu","hybrid","logging_config","ml_risk_predictor","multichannel",
                  "polars_loader","pre_processing","serialize","surrogate","user","visualization"]:
            __import__(f"havolib.{m}")
    def test_init_exports(self):
        import havolib; assert hasattr(havolib,"HavokEstimator")

class TestHavokEstimator:
    def test_fit_returns_self(self,clean_sine):
        from havolib.estimator import HavokEstimator; assert HavokEstimator(r=4,tau=2,m=15).fit(clean_sine) is not None
    def test_forcing_shape(self,clean_sine,fitted_estimator): assert fitted_estimator.forcing_.shape==clean_sine.shape
    def test_risk_shape(self,clean_sine,fitted_estimator): assert fitted_estimator.risk_.shape==clean_sine.shape
    def test_risk_binary(self,fitted_estimator): assert set(np.unique(fitted_estimator.risk_)).issubset({0,1})
    def test_sv_positive_descending(self,fitted_estimator):
        sv=fitted_estimator.singular_values_; assert np.all(sv>0); assert np.all(np.diff(sv)<=1e-9)
    def test_forcing_finite(self,fitted_estimator): assert np.all(np.isfinite(fitted_estimator.forcing_))
    def test_fit_transform_equals(self,noisy_sine):
        from havolib.estimator import HavokEstimator
        e1=HavokEstimator(r=4,tau=2,m=15,random_state=1); e2=HavokEstimator(r=4,tau=2,m=15,random_state=1)
        ft=e1.fit_transform(noisy_sine); e2.fit(noisy_sine); t=e2.get_forcing()
        np.testing.assert_array_almost_equal(ft,t,decimal=10)
    def test_all_diff_methods(self,clean_sine):
        from havolib.estimator import HavokEstimator
        for m in ["finite_diff","spline","total_variation","gradient"]:
            e=HavokEstimator(r=4,tau=2,m=15,diff_method=m).fit(clean_sine)
            assert np.all(np.isfinite(e.forcing_)),f"{m} failed"
    def test_auto_tau_m(self,clean_sine):
        from havolib.estimator import HavokEstimator
        e=HavokEstimator(r=4,tau="auto",m="auto").fit(clean_sine)
        assert isinstance(e.tau_fitted_,int) and e.tau_fitted_>=1
        assert isinstance(e.m_fitted_,int) and e.m_fitted_>=5
    def test_score_float(self,fitted_estimator,clean_sine): s=fitted_estimator.score(clean_sine); assert isinstance(s,float) and s>=0
    def test_get_set_params_roundtrip(self):
        from havolib.estimator import HavokEstimator
        e=HavokEstimator(r=7,tau=3,m=25,threshold_std=2.5); p=e.get_params()
        e2=HavokEstimator().set_params(**p)
        assert e2.r==7 and e2.tau==3 and e2.m==25
    def test_regime_shift_detected(self,regime_shift_signal):
        from havolib.estimator import HavokEstimator
        e=HavokEstimator(r=5,tau=2,m=20).fit(regime_shift_signal)
        assert np.sum(e.risk_[len(regime_shift_signal)//2:])>0
    def test_2d_input(self):
        from havolib.estimator import HavokEstimator
        e=HavokEstimator(r=3,tau=1,m=10).fit(np.random.default_rng(1).standard_normal((500,1)))
        assert e.forcing_.shape[0]==500
    def test_predict_risk_agrees(self,fitted_estimator,clean_sine):
        np.testing.assert_array_equal(fitted_estimator.predict_risk(clean_sine),fitted_estimator.risk_)

class TestEmbedding:
    def test_hankel_shape(self,clean_sine):
        from havolib.embedding import hankel_matrix
        H=hankel_matrix(clean_sine,m=30,tau=2); assert H.ndim==2; assert H.shape[1]==30
    def test_hankel_no_nan(self,clean_sine):
        from havolib.embedding import hankel_matrix; assert np.all(np.isfinite(hankel_matrix(clean_sine,m=20,tau=1)))
    def test_hankel_first_row(self,clean_sine):
        from havolib.embedding import hankel_matrix
        np.testing.assert_array_equal(hankel_matrix(clean_sine,m=5,tau=1)[0],clean_sine[:5])

class TestDecomposition:
    def test_shapes(self,clean_sine):
        from havolib.embedding import hankel_matrix; from havolib.decomposition import eigen_time_delay
        H=hankel_matrix(clean_sine,m=20,tau=2); V,s=eigen_time_delay(H,5)
        assert V.shape==(H.shape[0],5); assert s.shape==(5,)
    def test_sv_non_negative(self,clean_sine):
        from havolib.embedding import hankel_matrix; from havolib.decomposition import eigen_time_delay
        _,s=eigen_time_delay(hankel_matrix(clean_sine,m=20,tau=2),5); assert np.all(s>=0)
    def test_orthogonal(self,clean_sine):
        from havolib.embedding import hankel_matrix; from havolib.decomposition import eigen_time_delay
        H=hankel_matrix(clean_sine,m=20,tau=2); V,_=eigen_time_delay(H,5)
        np.testing.assert_allclose(V.T@V,np.eye(5),atol=0.1)

class TestDetection:
    def test_binary(self):
        from havolib.detection import threshold_risk
        assert set(np.unique(threshold_risk(np.random.default_rng(5).standard_normal(1000),50,3.0))).issubset({0,1})
    def test_shape(self): from havolib.detection import threshold_risk; assert threshold_risk(np.random.randn(500),50,2.0).shape==(500,)
    def test_zeros(self): from havolib.detection import threshold_risk; assert np.all(threshold_risk(np.zeros(500),50,2.0)==0)
    def test_spike(self):
        from havolib.detection import threshold_risk; f=np.zeros(1000); f[500]=1000
        assert np.sum(threshold_risk(f,50,2.0)[400:600])>0

class TestDifferentiation:
    @pytest.fixture
    def V(self): t=np.linspace(0,10,500); return np.column_stack([np.sin(t),np.cos(t),np.sin(2*t)]),t
    def test_shapes(self,V):
        from havolib.estimator import finite_diff,spline_diff,total_variation_diff
        Vv,t=V
        for fn in [finite_diff,spline_diff,total_variation_diff]: assert fn(Vv,t).shape==Vv.shape
    def test_spline_cos(self):
        """Spline derivative of sin(t) should approximate cos(t)."""
        from havolib.estimator import spline_diff
        t=np.linspace(0,2*np.pi,200); V=np.sin(t).reshape(-1,1)
        dv=spline_diff(V,t); expected=np.cos(t).reshape(-1,1)
        corr=np.corrcoef(dv[10:-10,0],expected[10:-10,0])[0,1]
        assert corr>0.7,f"Spline-cos correlation too low: {corr:.3f}"

class TestCrossValidation:
    def test_keys(self,clean_sine):
        from havolib.estimator import cross_val_score_havok
        r=cross_val_score_havok(clean_sine,{"r":[3,5],"tau":[2],"m":[15]},cv=2)
        assert "best_params" in r and "best_score" in r
    def test_count(self,clean_sine):
        from havolib.estimator import cross_val_score_havok
        r=cross_val_score_havok(clean_sine,{"r":[3,5,7],"tau":[1,2],"m":[15]},cv=2)
        assert len(r["cv_results"])==6

class TestSerialization:
    def test_save_load_roundtrip(self,fitted_estimator):
        from havolib.serialize import save_pipeline,load_pipeline
        with tempfile.NamedTemporaryFile(suffix='.havok',delete=False) as f: p=f.name
        try:
            save_pipeline(p,"0.7.1",fitted_estimator.get_params(),{"forcing":fitted_estimator.forcing_})
            l=load_pipeline(p); assert np.allclose(l["arrays"]["forcing"],fitted_estimator.forcing_)
        finally: os.unlink(p)

class TestEdgeCases:
    def test_constant(self):
        from havolib.estimator import HavokEstimator
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); e=HavokEstimator(r=3,tau=1,m=10).fit(np.ones(500))
        assert np.all(np.isfinite(e.forcing_))
    def test_zeros(self):
        from havolib.estimator import HavokEstimator
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); e=HavokEstimator(r=3,tau=1,m=10).fit(np.zeros(500))
        assert e.forcing_.shape[0]==500
    def test_spike(self):
        from havolib.estimator import HavokEstimator; x=np.zeros(500); x[250]=100
        e=HavokEstimator(r=3,tau=1,m=10).fit(x); assert np.all(np.isfinite(e.forcing_))
    def test_very_short(self):
        from havolib.estimator import HavokEstimator
        with pytest.raises(Exception): HavokEstimator(r=3,tau=1,m=5).fit(np.arange(10,dtype=float))
    def test_negative(self):
        from havolib.estimator import HavokEstimator
        e=HavokEstimator(r=4,tau=2,m=15).fit(-np.sin(np.linspace(0,20,1000))); assert np.all(np.isfinite(e.forcing_))
    def test_large_amp(self):
        from havolib.estimator import HavokEstimator
        e=HavokEstimator(r=4,tau=2,m=15).fit(np.sin(np.linspace(0,20,2000))*1e6); assert np.all(np.isfinite(e.forcing_))
    def test_prefit_error(self):
        from havolib.estimator import HavokEstimator
        with pytest.raises(Exception): HavokEstimator().transform()
        with pytest.raises(Exception): HavokEstimator().get_risk()

class TestIntegration:
    def test_full_chain(self,clean_sine):
        from havolib.estimator import HavokEstimator
        risk=HavokEstimator(r=5,tau=2,m=20).fit(clean_sine).get_risk()
        assert set(np.unique(risk.astype(int))).issubset({0,1})
    def test_pipeline_end_to_end(self,clean_sine):
        from havolib.pipeline import HavokPipeline
        pipe=HavokPipeline(tau=2,m=20,r=5).fit(np.arange(len(clean_sine)),clean_sine)
        assert np.all(np.isfinite(pipe.get_forcing()))

class TestRegression:
    REF_SEED=42; N=3000
    @pytest.fixture(scope="class")
    def ref(self):
        rng=np.random.default_rng(42); t=np.linspace(0,50,3000); return np.sin(t)+rng.normal(0,0.05,3000)
    def test_forcing_repro(self,ref):
        from havolib.estimator import HavokEstimator
        e1=HavokEstimator(r=5,tau=2,m=20,random_state=0).fit(ref)
        e2=HavokEstimator(r=5,tau=2,m=20,random_state=0).fit(ref)
        np.testing.assert_array_almost_equal(e1.forcing_,e2.forcing_,decimal=10)
    def test_risk_repro(self,ref):
        from havolib.estimator import HavokEstimator
        e1=HavokEstimator(r=5,tau=2,m=20,random_state=0).fit(ref)
        e2=HavokEstimator(r=5,tau=2,m=20,random_state=0).fit(ref)
        np.testing.assert_array_equal(e1.risk_,e2.risk_)

class TestStress:
    def test_50k(self):
        from havolib.estimator import HavokEstimator
        rng=np.random.default_rng(1); x=np.sin(np.linspace(0,100,50000))+rng.normal(0,0.1,50000)
        t0=time.time(); e=HavokEstimator(r=5,tau=2,m=20).fit(x)
        assert np.all(np.isfinite(e.forcing_)); assert time.time()-t0<120
    def test_batch_50(self):
        from havolib.estimator import HavokEstimator
        rng=np.random.default_rng(77)
        for i in range(50):
            x=np.sin(np.linspace(0,10,500))+rng.normal(0,0.1,500)
            e=HavokEstimator(r=3,tau=1,m=10).fit(x); assert np.all(np.isfinite(e.forcing_))
    def test_high_r(self):
        from havolib.estimator import HavokEstimator
        x=np.sin(np.linspace(0,50,5000))
        try: e=HavokEstimator(r=20,tau=2,m=50).fit(x); assert np.all(np.isfinite(e.forcing_))
        except (ValueError,np.linalg.LinAlgError): pass

class TestConfig:
    def test_yaml_loads(self):
        import yaml
        p=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","havok_config.yaml"))
        if os.path.exists(p):
            with open(p) as f: assert isinstance(yaml.safe_load(f),dict)
    def test_engine_yaml(self):
        import yaml
        p=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","engine.yaml"))
        if os.path.exists(p):
            with open(p) as f: assert isinstance(yaml.safe_load(f),dict)

class TestSklearnContract:
    def test_all_keys(self):
        from havolib.estimator import HavokEstimator
        for k in ["r","tau","m","threshold_std","window","diff_method","svd_solver"]:
            assert k in HavokEstimator(r=7,tau=3,m=25).get_params()
    def test_clone(self):
        from sklearn.base import clone; from havolib.estimator import HavokEstimator
        c=clone(HavokEstimator(r=6,tau=3,m=20)); assert c.r==6

class TestAutoTune:
    def test_tau_int(self,clean_sine):
        from havolib.auto_tune import optimal_tau_mi; t=optimal_tau_mi(clean_sine,max_lag=50)
        assert isinstance(t,(int,np.integer)) and t>=1
    def test_m_int(self,clean_sine):
        from havolib.auto_tune import optimal_m_fnn; m=optimal_m_fnn(clean_sine,tau=2,max_m=30)
        assert isinstance(m,(int,np.integer)) and m>=1
    def test_freq_ordering(self):
        from havolib.auto_tune import optimal_tau_mi
        t=np.linspace(0,50,3000); low=np.sin(2*np.pi*0.05*t); high=np.sin(2*np.pi*0.5*t)
        assert optimal_tau_mi(low,max_lag=100)>=optimal_tau_mi(high,max_lag=100)

class TestGPU:
    def test_fallback(self,clean_sine):
        from havolib.gpu import svd; from havolib.embedding import hankel_matrix
        H=hankel_matrix(clean_sine,m=20,tau=2); U,s,Vt=svd(H,r=5)
        assert np.all(np.isfinite(U)) and np.all(s>=0)
    def test_is_available(self):
        from havolib.gpu import is_gpu_available; assert isinstance(is_gpu_available(),bool)

class TestMathProps:
    def test_forcing_mean(self,fitted_estimator):
        f=fitted_estimator.forcing_[fitted_estimator.m_fitted_:]; assert abs(np.mean(f))<1.0
    def test_sv_energy(self,fitted_estimator):
        sv=fitted_estimator.singular_values_; assert sv[0]==max(sv)
    def test_forcing_std_less(self,clean_sine,fitted_estimator):
        m=fitted_estimator.m_fitted_
        assert np.std(fitted_estimator.forcing_[m:])<np.std(clean_sine[m:])
    def test_risk_rate_clean(self,fitted_estimator): assert np.mean(fitted_estimator.risk_)<0.5
    def test_r_controls_rank(self,clean_sine):
        from havolib.estimator import HavokEstimator
        e3=HavokEstimator(r=3,tau=2,m=20).fit(clean_sine)
        e7=HavokEstimator(r=7,tau=2,m=20).fit(clean_sine)
        assert len(e3.singular_values_)<=len(e7.singular_values_)

class TestPackage:
    def test_pyproject(self):
        p=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","pyproject.toml")); assert os.path.exists(p)
    def test_requirements(self):
        p=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","requirements.txt"))
        assert os.path.exists(p); assert os.path.getsize(p)>10
