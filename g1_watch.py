"""Print the last live-monitor lines: python g1_watch.py logs/g1s_C4_live.jsonl [n]"""
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for d in L[-n:]:
    print(f"step {d['step']} ep_len {d['ep_len_s']:.1f}s | up {d['up']:+.2f} legs {d['legs_dir']:+.2f} pelvis_z {d['pelvis_z']:.2f} | contacts L_elb {d['L_elbow']:.2f} L_wr {d['L_wrist']:.2f} R_elb {d['R_elbow']:.2f} R_wr {d['R_wrist']:.2f} gate {d['gate']:.2f}")
    print("   " + "  ".join(f"{k[2:]}={v:.2f}" for k, v in d.items() if k.startswith("r_")))
