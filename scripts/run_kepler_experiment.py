#!/usr/bin/env python3
"""scripts/run_kepler_experiment.py — 端到端 kepler 模块实验"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LYNCEUS_DBG", "1")

def banner(m): print(f"\n{'='*60}\n  {m}\n{'='*60}")

def main():
    t0 = time.time(); R = {}

    banner("M121: kepler_losses")
    from lynceus.integrations.kepler_losses import mse_loss, log_mse_loss, huber_loss, asymmetric_loss
    yt, yp = np.array([10.,20.,5.,15.]), np.array([12.,18.,7.,14.])
    vals = {n: float(f(yt,yp)) for n,f in [("mse",mse_loss),("log_mse",log_mse_loss),("huber",lambda a,b:huber_loss(a,b,delta=1.5)),("asym",lambda a,b:asymmetric_loss(a,b,alpha_over=2.0,alpha_under=1.0))]}
    print(f"  {vals}"); R["loss"] = vals; assert vals["mse"]>0; print("  ✓ OK")

    banner("M122: kepler_mlp_base")
    from lynceus.integrations.kepler_mlp_base import NumpyMLP
    mlp = NumpyMLP(layer_dims=[8,16,8,4], activation="relu", dropout_rate=0.1)
    X = np.random.randn(32,8).astype(np.float32)
    out = mlp.forward(X, training=False)
    print(f"  {X.shape}→{out.shape} mean={out.mean():.4f}"); R["mlp"]={"shape":list(out.shape)}; assert out.shape==(32,4); print("  ✓ OK")

    banner("M123: kepler_training_prep")
    from lynceus.integrations.kepler_training_prep import get_np_type, normalize, one_hot
    print(f"  float→{get_np_type('float')}")
    normed = normalize(np.random.randn(50,3).astype(np.float32))[0]
    print(f"  normalized mean≈{normed.mean(axis=0).round(6)}")
    ohe = one_hot(np.array([0,2,1,3,0]), num_classes=4)
    print(f"  one_hot {ohe.shape}"); R["util"]={"ohe":list(ohe.shape)}; print("  ✓ OK")

    banner("M124: kepler_workload")
    from lynceus.integrations.kepler_workload import WorkloadGenerator, KeplerPlanDiscoverer
    synth = {"q1": {f"p{i}": {"results":[{"plan_id":j,"latency_ms":float(np.random.exponential(50)+j*10)} for j in range(5)]} for i in range(50)}}
    disc = KeplerPlanDiscoverer(query_execution_data=synth); print(f"  plans={disc.plan_ids}")
    gen = WorkloadGenerator(synth); wl = gen.full_workload()
    print(f"  workload: {len(wl)} instances"); R["wl"]={"n":len(wl)}; print("  ✓ OK")

    banner("M125: kepler_db_simulator")
    from lynceus.integrations.kepler_db_simulator import PlannedQuery, DatabaseSimulator
    sim = DatabaseSimulator(synth, noise_sigma=0.05)
    lats = [sim.execute(PlannedQuery(query_id="q1",plan_id=0,parameters=[f"p{i}"])) for i in range(10)]
    print(f"  10 queries: mean={np.mean(lats):.1f} std={np.std(lats):.1f}")
    R["sim"]={"mean_lat":float(np.mean(lats))}; print("  ✓ OK")

    banner("M126: kepler_trainer")
    from lynceus.integrations.kepler_trainer import ClassificationTrainer
    meta = {"predicates":[{"data_type":"float","name":f"f{i}"} for i in range(4)]}
    clf = ClassificationTrainer(metadata=meta, plan_ids=list(range(5)), input_dim=4, rng_seed=42)
    Xt = np.random.randn(80,4).astype(np.float32); yt = np.argmin(np.random.exponential(50,(80,5)),axis=1)
    hist = clf.train(Xt, yt, epochs=30, batch_size=16, lr=0.01)
    fl = hist.history["loss"][-1]; preds = clf.predict(Xt[:5])
    print(f"  loss={fl:.4f} preds={preds}"); R["trainer"]={"loss":float(fl)}; print("  ✓ OK")

    banner(f"全部通过 ({time.time()-t0:.1f}s)")
    print(json.dumps(R, indent=2, default=str))
    os.makedirs("output", exist_ok=True)
    with open("output/kepler_experiment.json","w") as f: json.dump(R,f,indent=2,default=str)
    print("  → output/kepler_experiment.json")

if __name__=="__main__": main()
