"""
patch_gcn.py - minimal instrumentation of the GCN DR repo (github.com/mfar201/diabetic_retinopathy_classification_gcn,
commit fb123ab), PLOS Comput Biol 2025.

Training, model, transforms, loss, optimiser, scheduler, early stopping and the released checkpoint rule are unchanged,
including two behaviours of the released code that we measure rather than fix:
  * best_model_state = model.state_dict() keeps references, so the "best validation" model restored before testing
    is in fact the last-epoch model;
  * test (and validation) batches come from ImageFolder in class-sorted order, and the model builds its k-NN graph
    across the images of each batch.
Added:
  1. Settings from env: backbone (LEAK_BACKBONE, LEAK_OUTDIM), data folders (LEAK_TRAIN/VAL/TEST), seed (LEAK_SEED),
     max epochs (LEAK_EPOCHS, smoke tests only), output folder (LEAK_OUT).
  2. Per-epoch validation and test predictions -> LEAK_OUT/evals/eXXX_{val,test}.npz (test is never used for any decision).
  3. After the released test evaluation: the same model re-evaluated on the test set in 5 random orders and one image at
     a time (no cross-image graph), and the true best-validation weights (a deep copy) evaluated the same ways.
  4. Compatibility only: ReduceLROnPlateau(verbose=True) removed (the argument only printed messages; it was removed in
     recent PyTorch).
Usage: python patch_gcn.py /path/to/repo
"""
import sys
from pathlib import Path
root = Path(sys.argv[1])

def rep(path, old, new):
    p = root / path; s = p.read_text()
    if (new and new in s) or (not new and old not in s): return   # already applied
    assert s.count(old) == 1, (path, old[:80])
    p.write_text(s.replace(old, new, 1))

# 1. settings from env
rep("config.py", """        'models': [
            {
                'name': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
                'output_dim': 768,
                'type': 'transformer'
            }
        ],""", """        'models': [
            {
                'name': __import__('os').environ.get('LEAK_BACKBONE', 'vit_base_patch16_224.augreg2_in21k_ft_in1k'),  # LEAK_PATCH
                'output_dim': int(__import__('os').environ.get('LEAK_OUTDIM', 768)),
                'type': 'transformer'
            }
        ],""")
rep("config.py", "        'num_epochs': 50,", "        'num_epochs': int(__import__('os').environ.get('LEAK_EPOCHS', 50)),  # LEAK_PATCH")
rep("config.py", """        'train_dir': '',
        'val_dir': '',
        'test_dir': '',""", """        'train_dir': __import__('os').environ.get('LEAK_TRAIN', ''),  # LEAK_PATCH
        'val_dir': __import__('os').environ.get('LEAK_VAL', ''),
        'test_dir': __import__('os').environ.get('LEAK_TEST', ''),""")

# 2. keep predictions of the last evaluate() call
rep("utils/training.py", """    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(val_loader)
""", """    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(val_loader)
    evaluate.last = (all_probs, all_labels)  # LEAK_PATCH
""")

# 3. seed from env
rep("train.py", """    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
""", """    _leak_seed = int(os.environ.get('LEAK_SEED', 42))  # LEAK_PATCH (released: 42)
    torch.manual_seed(_leak_seed)
    torch.cuda.manual_seed_all(_leak_seed)
    np.random.seed(_leak_seed)
    import random as _random; _random.seed(_leak_seed)
""")
rep("train.py", "            verbose=True,\n", "")

# 4. per-epoch dumps (test predictions are recorded only; nothing reads them during training)
rep("train.py", """            val_metrics = evaluate(model, val_loader, criterion, device, class_names=class_names)
""", """            val_metrics = evaluate(model, val_loader, criterion, device, class_names=class_names)
            # ---- LEAK_PATCH: record validation + test predictions for this epoch ----
            _leak_out = os.environ.get('LEAK_OUT', '.'); os.makedirs(os.path.join(_leak_out, 'evals'), exist_ok=True)
            _p, _y = evaluate.last
            np.savez_compressed(os.path.join(_leak_out, 'evals', 'e%03d_val.npz' % epoch), prob=_p.astype(np.float32), y=_y)
            evaluate(model, test_loader, criterion, device, class_names=class_names); _p, _y = evaluate.last
            np.savez_compressed(os.path.join(_leak_out, 'evals', 'e%03d_test.npz' % epoch), prob=_p.astype(np.float32), y=_y)
""")
rep("train.py", """                best_model_state = model.state_dict()
""", """                best_model_state = model.state_dict()
                _leak_best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; _leak_best_epoch = epoch  # LEAK_PATCH
""")

# 5. after the released test evaluation: order / batching checks and the true best-validation weights
rep("train.py", """        print("\\nFinal Test Metrics:")
""", """        # ---- LEAK_PATCH: released result, then the same model under other test orderings, then true best-val weights ----
        import copy as _copy, json as _json
        _leak_out = os.environ.get('LEAK_OUT', '.')
        _p, _y = evaluate.last
        np.savez_compressed(os.path.join(_leak_out, 'final_released_sorted.npz'), prob=_p.astype(np.float32), y=_y)
        import pandas as _pd
        _pd.DataFrame(test_dataset.samples, columns=['path', 'label']).to_csv(os.path.join(_leak_out, 'test_files.csv'), index=False)
        def _leak_orders(tag):
            g = np.random.RandomState(12345)
            for k in range(5):
                perm = g.permutation(len(test_dataset))
                dl = torch.utils.data.DataLoader(torch.utils.data.Subset(test_dataset, perm.tolist()), batch_size=config['batch_size'],
                                                 shuffle=False, num_workers=4)
                evaluate(model, dl, criterion, device); pp, yy = evaluate.last
                np.savez_compressed(os.path.join(_leak_out, '%s_perm%d.npz' % (tag, k)), prob=pp.astype(np.float32), y=yy, perm=perm)
            dl = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
            evaluate(model, dl, criterion, device); pp, yy = evaluate.last
            np.savez_compressed(os.path.join(_leak_out, '%s_bs1.npz' % tag), prob=pp.astype(np.float32), y=yy)
        _leak_orders('final_released')
        _stop_epoch = epoch
        if best_model_state is not None:
            _rel = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(_leak_best)
            evaluate(model, test_loader, criterion, device); pp, yy = evaluate.last
            np.savez_compressed(os.path.join(_leak_out, 'true_bestval_sorted.npz'), prob=pp.astype(np.float32), y=yy)
            _leak_orders('true_bestval')
            model.load_state_dict(_rel)  # back to the released model before the released code saves it
        _json.dump(dict(stop_epoch=int(_stop_epoch), true_best_val_epoch=int(_leak_best_epoch), best_val_f1=float(best_f1),
                        n_train=len(train_dataset), n_val=len(val_dataset), n_test=len(test_dataset)),
                   open(os.path.join(_leak_out, 'leak_meta.json'), 'w'))
        open(os.path.join(_leak_out, 'LEAK_DONE'), 'w').write('ok')
        print("\\nFinal Test Metrics:")
""")
print("patched", root)
