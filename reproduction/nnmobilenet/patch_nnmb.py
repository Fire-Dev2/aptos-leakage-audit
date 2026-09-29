"""
patch_nnmb.py — minimal instrumentation of nnMobileNet (github.com/Retinal-Research/NN-MOBILENET, commit 920acd3).

Training, augmentation, optimiser, EMA and the released checkpoint rule (keep the epoch with the best
TEST score, evaluated on the test set every epoch) are unchanged. Added:
  1. APTOS paths from env (LEAK_IMG_DIR, LEAK_TRAIN_CSV, LEAK_TEST_CSV) — the authors' split files
     (dataset/APOTS/train_1.csv, test_1.csv) and cropped images were not released.
  2. Per-epoch TEST predictions of the model and of its EMA copy -> <output_dir>/evals/eXXXX_test_{model,ema}.npz
  3. If LEAK_VAL_CSV is set: the same evaluation on a held-out validation set -> eXXXX_val_{model,ema}.npz
  4. Pretrained ReXNet-3.0 weights: loaded as released (strict=False), plus a check that the backbone
     actually loaded (the released code silently ignores mismatched keys).
Usage: python patch_nnmb.py /path/to/NN-MOBILENET
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])

def rep(path, old, new):
    p = root / path; s = p.read_text()
    if new in s:          # already applied
        return
    assert s.count(old) == 1, (path, old[:70])
    p.write_text(s.replace(old, new, 1))

# 1. dataset paths
rep("dataset.py",
    """            dataset = Apots(image_dir='dataset/APOTS/crop',label_dir='dataset/APOTS/train_1.csv',transform=transform)
        else:
            dataset = Apots(image_dir='dataset/APOTS/crop',label_dir='dataset/APOTS/test_1.csv',transform=transform)""",
    """            dataset = Apots(image_dir=os.environ.get('LEAK_IMG_DIR', 'dataset/APOTS/crop'),  # LEAK_PATCH
                            label_dir=os.environ.get('LEAK_TRAIN_CSV', 'dataset/APOTS/train_1.csv'), transform=transform)
        else:
            dataset = Apots(image_dir=os.environ.get('LEAK_IMG_DIR', 'dataset/APOTS/crop'),
                            label_dir=os.environ.get('LEAK_TEST_CSV', 'dataset/APOTS/test_1.csv'), transform=transform)""")

# 2. keep the probabilities of the last evaluation
rep("engine.py", "    preds = np.argmax(probs, axis=1)\n",
    "    preds = np.argmax(probs, axis=1)\n    evaluate.last = (probs, gts)  # LEAK_PATCH\n")

# 3. pretrained weights, checked
rep("main.py", "    model.load_state_dict(torch.load('rexnet_3.0.pth'),strict=False)\n",
    """    # ---- LEAK_PATCH: same strict=False load as released, but verify the backbone loaded ----
    _pt = os.environ.get('LEAK_PRETRAIN', 'rexnet_3.0.pth')
    if os.path.exists(_pt):
        _sd = torch.load(_pt, map_location='cpu', weights_only=True)
        _sd = _sd.get('state_dict', _sd) if isinstance(_sd, dict) else _sd
        _own = model.state_dict()
        _ok = [k for k in _sd if k in _own and _own[k].shape == _sd[k].shape]
        print('LEAK_PATCH pretrained: %d/%d model tensors matched from %s' % (len(_ok), len(_own), _pt))
        if len(_ok) < 0.9 * sum(1 for k in _own if k.startswith('features')): print('LEAK_PATCH WARNING: backbone only partly matched the pretrained file (released code would do the same silently)')
        model.load_state_dict({k: _sd[k] for k in _ok}, strict=False)
    elif os.environ.get('LEAK_ALLOW_NO_PRETRAIN'):
        print('LEAK_PATCH WARNING: no pretrained weights (smoke test only)')
    else:
        raise FileNotFoundError(_pt)
""")

# 4. validation loader + prediction dumps
rep("main.py", "    mixup_fn = None\n",
    """    # ---- LEAK_PATCH: prediction dumps and optional validation set ----
    import os as _os, time as _time
    from dataloader.Apots import Apots as _Apots
    from dataset import build_transform as _bt
    _leak_dir = _os.path.join(args.output_dir, 'evals'); _os.makedirs(_leak_dir, exist_ok=True)
    _leak_val = None
    if _os.environ.get('LEAK_VAL_CSV'):
        _vds = _Apots(image_dir=_os.environ['LEAK_IMG_DIR'], label_dir=_os.environ['LEAK_VAL_CSV'], transform=_bt(False, args))
        _leak_val = torch.utils.data.DataLoader(_vds, sampler=torch.utils.data.SequentialSampler(_vds),
                                                batch_size=int(1.5 * args.batch_size), num_workers=args.num_workers,
                                                pin_memory=args.pin_mem, drop_last=False)
        print('LEAK_PATCH: validation set with', len(_vds), 'images')
    def _leak_dump(tag, epoch):
        _p, _y = evaluate.last
        np.savez_compressed(_os.path.join(_leak_dir, 'e%04d_%s.npz' % (epoch, tag)), prob=_p.astype(np.float32), y=_y)
    _leak_t0 = _time.time()

    mixup_fn = None
""")
rep("main.py",
    """            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"kaapa of the model""",
    """            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            _leak_dump('test_model', epoch)  # LEAK_PATCH
            if _leak_val is not None:
                evaluate(_leak_val, model, device, use_amp=args.use_amp); _leak_dump('val_model', epoch)
            print(f"kaapa of the model""")
rep("main.py",
    """                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)\n""",
    """                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                _leak_dump('test_ema', epoch)  # LEAK_PATCH
                if _leak_val is not None:
                    evaluate(_leak_val, model_ema.ema, device, use_amp=args.use_amp); _leak_dump('val_ema', epoch)
                with open(_os.path.join(_leak_dir, 'timing.csv'), 'a') as _fh:
                    _fh.write('%d,%.1f\\n' % (epoch, _time.time() - _leak_t0))
""")
# 5. compatibility (no change in results): our own resume checkpoints contain argparse/numpy objects,
#    which torch>=2.6 refuses to load by default
rep("utils.py", "            checkpoint = torch.load(args.resume, map_location='cpu')\n",
    "            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)  # LEAK_PATCH compat\n")
print("patched", root)
