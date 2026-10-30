"""Embed codon / amino-acid sequences with GenSLM-ESM and extract SAE features on XPU.

Usage:
  python embed_and_sae.py --out OUTDIR [--sae-dir CKPT/layerNN] [--layer 32]
"""
import argparse, json, time
from pathlib import Path

import torch
from genslm_esm_sae.backbone import Backbone
from genslm_esm_sae.sae import SaeModel

MODEL = 'genslm-test/genslm-esmc-600M-contrastive-2026-10-step-200000'
SAE_DEFAULT = ('/flare/FRAME-IDP/msinclair/software/genslm-esm-sae/campaigns/'
               '600m-codon/runs/l18-36-lr2e-4/best/layer32')

# Codon input: space-separated uppercase codons. Amino acid input: space-separated residues.
CODON = ['ATG GCT AAA GGT GAA CTG TTC ACC GGT GTT GTG CCC ATC CTG GTC GAG CTG GAC GGC GAC TAA',
         'ATG AGC AAA GGA GAA GAA CTT TTC ACT GGA GTT GTC CCA ATT CTT GTT GAA TTA GAT TAA']
AA = ['M A K G E L F T G V V P I L V E L D G D',
      'M S K G E E L F T G V V P I L V E L D']


def to_dev(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}


def embed(bb, seqs, modality, dev):
    batch = to_dev(bb.collator([{modality: s} for s in seqs]), dev)
    acts = bb(batch)  # (n_real_tokens, d_model) or (n_layers, n_tokens, d)
    # per-sequence mean pool over real tokens
    real = batch[f'{modality}_labels'] != -100
    counts = real.sum(1).tolist()
    a = acts if acts.dim() == 2 else acts[-1]
    pooled = torch.stack([c.mean(0) for c in torch.split(a, counts)])
    return batch, acts, pooled, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sae-dir', default=SAE_DEFAULT)
    p.add_argument('--layer', type=int, default=32)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    print('device', dev, torch.xpu.get_device_name(0) if dev.type == 'xpu' else '')
    res = {'device': str(dev)}

    t = time.time()
    bb = Backbone(MODEL, layer=[1, a.layer], modality='codon', max_length=2048,
                  dtype=torch.bfloat16)
    bb.model.to(dev)
    res['load_s'] = time.time() - t
    print('n_layers', bb.model.config.n_layers, 'd_model', bb.model.config.d_model)

    t = time.time()
    _, acts, pooled, counts = embed(bb, CODON, 'codon', dev)
    res['codon'] = {'acts_shape': list(acts.shape), 'pooled_shape': list(pooled.shape),
                    'tokens_per_seq': counts, 'finite': bool(torch.isfinite(acts).all()),
                    'cos_sim': float(torch.nn.functional.cosine_similarity(pooled[0], pooled[1], 0))}
    torch.save(pooled.cpu(), a.out / ('codon_layer%02d_meanpool.pt' % a.layer))
    print('codon', res['codon'])

    # Amino-acid modality reuses the same weights; just swap the collator.
    bb_aa = Backbone.__new__(Backbone)
    bb_aa.__dict__.update(bb.__dict__)
    from genslm_esm.data import GenslmEsmcDataCollator
    bb_aa.modality = 'aminoacid'
    bb_aa.collator = GenslmEsmcDataCollator(return_codon=False, return_aminoacid=True,
                                            max_length=2048, tokenizer=bb.collator.tokenizer)
    _, acts_aa, pooled_aa, counts_aa = embed(bb_aa, AA, 'aminoacid', dev)
    res['aminoacid'] = {'acts_shape': list(acts_aa.shape), 'tokens_per_seq': counts_aa,
                        'finite': bool(torch.isfinite(acts_aa).all())}
    torch.save(pooled_aa.cpu(), a.out / ('aa_layer%02d_meanpool.pt' % a.layer))
    print('aminoacid', res['aminoacid'])
    res['embed_s'] = time.time() - t

    # SAE feature extraction on the layer the SAE was trained on.
    sae = SaeModel.from_pretrained(a.sae_dir).eval().to(dev)
    assert sae.config.layer == a.layer, (sae.config.layer, a.layer)
    with torch.inference_mode():
        out = sae(acts[-1])  # (n_tokens, codebook_dim)
    fm = out.feature_magnitudes
    nz = (fm != 0).sum(-1)
    top = torch.topk(fm.sum(0), 10)
    res['sae'] = {'config': {k: getattr(sae.config, k) for k in
                             ('backbone_id', 'layer', 'modality', 'd_model', 'codebook_dim', 'k')},
                  'features_shape': list(fm.shape),
                  'active_per_token_max': int(nz.max()),
                  'mean_recon_loss': float(out.reconstruction_loss.mean()),
                  'top10_features': top.indices.tolist()}
    torch.save({'feature_magnitudes': fm.to_sparse().cpu(), 'tokens_per_seq': counts},
               a.out / ('sae_layer%02d_features.pt' % a.layer))
    print('sae', res['sae'])
    if dev.type == 'xpu':
        res['xpu_max_mem_GB'] = torch.xpu.max_memory_allocated() / 1e9
    (a.out / 'results.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
