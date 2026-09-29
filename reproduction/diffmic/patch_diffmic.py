"""
patch_diffmic.py — minimal, behaviour-preserving instrumentation of DiffMIC (commit 8a094f7).

What it adds (nothing else changes; ckpt_best.pth is still chosen on TEST accuracy, as released):
  1. At every evaluation, saves per-image TEST predictions to <log_path>/evals/eXXXX_test.npz
  2. If env LEAK_VAL_PKL is set, also evaluates a VALIDATION set with the identical procedure
     and saves <log_path>/evals/eXXXX_val.npz
  3. Appends wall-clock timing per evaluation to <log_path>/evals/timing.csv

Usage:  python patch_diffmic.py /path/to/DiffMIC
"""
import sys, re
from pathlib import Path

root = Path(sys.argv[1])
f = root / "diffusion_trainer.py"
s = f.read_text()
if "LEAK_PATCH" in s:
    print("already patched"); sys.exit(0)

HELPER = '''
    # ---- LEAK_PATCH: helper that mirrors the released evaluation loop exactly ----
    def _leak_predict(self, model, loader, config):
        y1_pred, y1_true = None, None
        for images, target in loader:
            images_unflat = images.to(self.device)
            if config.data.dataset == "toy" or config.model.arch == "simple" or config.model.arch == "linear":
                images = torch.flatten(images, 1)
            images = images.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
                target_pred = target_pred.softmax(dim=1)
                y_T_mean = target_pred
                if config.diffusion.noise_prior:
                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                if not config.diffusion.noise_prior:
                    target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
                    target_pred = target_pred.softmax(dim=1)
                label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean, self.num_timesteps, self.alphas,
                                          self.one_minus_alphas_bar_sqrt, only_last_sample=True)
            y1_pred = torch.cat([y1_pred, label_t_0]) if y1_pred is not None else label_t_0
            y1_true = torch.cat([y1_true, target]) if y1_true is not None else target
        return y1_pred.detach().cpu().numpy(), y1_true.cpu().numpy()

    def train(self):'''
assert s.count("    def train(self):") == 1
s = s.replace("    def train(self):", HELPER, 1)

LOADERS = '''print('successfully load')
        # ---- LEAK_PATCH: optional validation loader + prediction dump dir ----
        import os as _os, time as _time
        _leak_dir = _os.path.join(self.args.log_path, "evals"); _os.makedirs(_leak_dir, exist_ok=True)
        _leak_val = None
        if _os.environ.get("LEAK_VAL_PKL"):
            _leak_val = data.DataLoader(APTOSDataset(data_list=_os.environ["LEAK_VAL_PKL"], train=False),
                                        batch_size=config.testing.batch_size, shuffle=False,
                                        num_workers=config.data.num_workers)
            print("LEAK_PATCH: validation set with", len(_leak_val.dataset), "images")
        _leak_t0 = _time.time()'''
assert s.count("print('successfully load')") == 1
s = s.replace("print('successfully load')", LOADERS, 1)

DUMP = '''kappa_avg = cohen_kappa(y1_pred.detach().cpu(), y1_true.cpu()).item()
                        # ---- LEAK_PATCH: dump test predictions, then evaluate validation set ----
                        np.savez_compressed(_os.path.join(_leak_dir, "e%04d_test.npz" % epoch),
                                            prob=y1_pred.detach().cpu().numpy(), y=y1_true.cpu().numpy())
                        if _leak_val is not None:
                            _vp, _vy = self._leak_predict(model, _leak_val, config)
                            np.savez_compressed(_os.path.join(_leak_dir, "e%04d_val.npz" % epoch), prob=_vp, y=_vy)
                        with open(_os.path.join(_leak_dir, "timing.csv"), "a") as _fh:
                            _fh.write("%d,%.1f\\n" % (epoch, _time.time() - _leak_t0))'''
assert s.count("kappa_avg = cohen_kappa(y1_pred.detach().cpu(), y1_true.cpu()).item()") == 1
s = s.replace("kappa_avg = cohen_kappa(y1_pred.detach().cpu(), y1_true.cpu()).item()", DUMP, 1)

# ---- speed-up with identical results: reuse the image encoding across the 1000 sampling steps ----
# In eval mode encoder_x(x) is a deterministic function of the image, yet the released sampler recomputes it
# at every one of the 1000 reverse-diffusion steps. We compute it once per batch and hand the same tensor back.
# Random-number draws are unchanged (the encoder draws none). Disable with env LEAK_NO_ENC_CACHE=1.
ENC_CACHE = '''from diffusion_utils import *
# ---- LEAK_PATCH: encoder cache for evaluation ----
import os as _leak_os
class _LeakCachedEnc(torch.nn.Module):
    def __init__(self, feat):
        super().__init__(); self.feat = feat
    def forward(self, x):
        assert x.shape[0] == self.feat.shape[0]
        return self.feat
_leak_orig_p_sample_loop = p_sample_loop
def p_sample_loop(model, x, *args, **kwargs):
    enc = model.encoder_x
    if _leak_os.environ.get("LEAK_NO_ENC_CACHE") or enc.training or torch.is_grad_enabled():
        return _leak_orig_p_sample_loop(model, x, *args, **kwargs)
    model.encoder_x = _LeakCachedEnc(enc(x))
    try:
        return _leak_orig_p_sample_loop(model, x, *args, **kwargs)
    finally:
        model.encoder_x = enc
'''
assert s.count("from diffusion_utils import *\n") == 1
s = s.replace("from diffusion_utils import *\n", ENC_CACHE, 1)

# ---- pause / resume across time-limited sessions (Kaggle: 12 h per commit) ----
# At an evaluation epoch, if the next 10-epoch block would not finish before LEAK_DEADLINE (unix time),
# the full training state (both models, both optimizers, EMA, epoch/step, best-so-far, all RNG states)
# is written to <log_path>/resume.pth and the process exits with code 3. With LEAK_RESUME=1 (and
# --resume_training so main.py keeps the log folder) training continues from that exact point.
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new, 1)

rep("        _leak_t0 = _time.time()",
    '''        _leak_t0 = _time.time()
        _leak_resume_path = _os.path.join(self.args.log_path, "resume.pth")
        _leak_resume = bool(_os.environ.get("LEAK_RESUME")) and _os.path.exists(_leak_resume_path)
        if _leak_resume: print("LEAK_PATCH: will resume from", _leak_resume_path)''')

rep("            else:  # pre-train the guidance auxiliary classifier",
    '''            elif _leak_resume:  # LEAK_PATCH: guidance classifier state is restored from resume.pth below
                pass
            else:  # pre-train the guidance auxiliary classifier''')

rep('''            start_epoch, step = 0, 0
            if self.args.resume_training:''',
    '''            start_epoch, step = 0, 0
            _leak_max_acc = 0.0
            if _leak_resume:  # ---- LEAK_PATCH: full-state resume ----
                import random as _random
                _R = torch.load(_leak_resume_path, map_location=self.device, weights_only=False)
                model.load_state_dict(_R["model"]); optimizer.load_state_dict(_R["opt"])
                if ema_helper is not None: ema_helper.load_state_dict(_R["ema"])
                self.cond_pred_model.load_state_dict(_R["aux"]); aux_optimizer.load_state_dict(_R["aux_opt"])
                start_epoch, step, _leak_max_acc = _R["epoch"], _R["step"], _R["max_acc"]
                torch.set_rng_state(_R["rng_cpu"])
                if _R["rng_cuda"] is not None: torch.cuda.set_rng_state(_R["rng_cuda"])
                np.random.set_state(_R["rng_np"]); _random.setstate(_R["rng_py"])
                del _R
                logging.info("LEAK_PATCH: resumed at epoch %d, step %d" % (start_epoch, step))
            _leak_tblock = _time.time(); _leak_blkmax = float(_os.environ.get("LEAK_BLOCK_SEC", "1200"))
            if self.args.resume_training and not _leak_resume:''')

rep("            max_accuracy = 0.0\n", "            max_accuracy = _leak_max_acc\n")

rep('''                                    f"Max accuracy: {max_accuracy:.2f}%"
                            )
                        )
''', '''                                    f"Max accuracy: {max_accuracy:.2f}%"
                            )
                        )
                        # ---- LEAK_PATCH: pause cleanly before the session time limit (resumable) ----
                        _now = _time.time(); _leak_blkmax = max(_leak_blkmax, _now - _leak_tblock); _leak_tblock = _now
                        _leak_pause = bool(_os.environ.get("LEAK_DEADLINE")) and epoch + 1 < self.config.training.n_epochs \\
                                and _now + 1.25 * _leak_blkmax > float(_os.environ["LEAK_DEADLINE"])
                        _leak_ckpt = bool(_os.environ.get("LEAK_CKPT_EVERY_EVAL")) and epoch + 1 < self.config.training.n_epochs
                        if _leak_pause or _leak_ckpt:
                            import random as _random
                            torch.save(dict(model=model.state_dict(), opt=optimizer.state_dict(),
                                            ema=ema_helper.state_dict() if ema_helper is not None else None,
                                            aux=self.cond_pred_model.state_dict(), aux_opt=aux_optimizer.state_dict(),
                                            epoch=epoch + 1, step=step, max_acc=max_accuracy,
                                            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                                            rng_np=np.random.get_state(), rng_py=_random.getstate()),
                                       _leak_resume_path + ".tmp")
                            _os.replace(_leak_resume_path + ".tmp", _leak_resume_path)
                            if _leak_pause:
                                logging.info("LEAK_PATCH: paused after epoch %d; resume.pth saved" % epoch)
                                raise SystemExit(3)
''')

# ---- compatibility fixes (no change in computed values) ----
# scikit-learn >= 1.6 returns a Python float from cohen_kappa_score / f1_score, which has no .item()
for old, new in [
    ("kappa_avg = cohen_kappa(y1_pred.detach().cpu(), y1_true.cpu()).item()",
     "kappa_avg = float(cohen_kappa(y1_pred.detach().cpu(), y1_true.cpu()))"),
    ("f1_avg = compute_f1_score(y1_true,y1_pred).item()", "f1_avg = float(compute_f1_score(y1_true,y1_pred))"),
    ("kappa_avg += cohen_kappa(label_t_0.detach().cpu(), target.cpu()).item()",
     "kappa_avg += float(cohen_kappa(label_t_0.detach().cpu(), target.cpu()))"),
]:
    s = s.replace(old, new)

f.write_text(s)
print("patched", f)
