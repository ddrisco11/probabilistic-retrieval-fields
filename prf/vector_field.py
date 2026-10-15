"""Aligned small-to-base displacement geometry with train-only PCA."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .retrieval import metrics, paired_bootstrap


def align(base_train, small_train):
    """Rectangular orthogonal Procrustes, with translation but no scale fit."""
    bm, sm = base_train.mean(0), small_train.mean(0)
    u, _, vh = torch.linalg.svd((base_train-bm).T @ (small_train-sm), full_matrices=False)
    return u @ vh, bm, sm


def pca(x):
    mean = x.mean(0)
    eigenvalues, axes = torch.linalg.eigh((x-mean).T @ (x-mean))
    return mean, axes.flip(1), eigenvalues.flip(0).clamp_min(0)


def energy_ratio(x, axes, dims):
    return float((x @ axes[:, :dims]).square().sum()/x.square().sum().clamp_min(1e-20))


def official_splits(data, cache):
    import pyarrow.parquet as pq
    root=Path(cache)/'hub/datasets--microsoft--wiki_qa/snapshots'/data['metadata']['dataset_revision']
    qmap={q:i for i,q in enumerate(data['query_ids'])}
    result={}
    for split in ['train','validation','test']:
        ids=set()
        for relative, expected in data['metadata']['source_sha256'].items():
            if not Path(relative).name.startswith(split+'-'): continue
            path=root/relative
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
            ids.update(qmap[r['question_id']] for r in pq.read_table(path,columns=['question_id']).to_pylist())
        result[split]=sorted(ids)
    return result


def sample_statistics(small, base, splits, w, bm, sm):
    # Each endpoint is the same text under another encoder, not a relevant Q/A pair.
    train, valid, test = [splits[k] for k in ['train','validation','test']]
    aligned=(base-bm)@w+sm
    delta=aligned-small
    ep_mean, ep_axes, ep_eigen = pca(torch.cat([small[train],aligned[train]]))
    d_mean, d_axes, d_eigen = pca(delta[train])
    dt=delta[test]
    result={'mean_displacement_norm':float(dt.norm(dim=-1).mean()),
            'rms_displacement':float(dt.square().sum(-1).mean().sqrt()),
            'global_drift_energy_fraction':float(dt.mean(0).square().sum()/dt.square().sum(-1).mean()),
            'base_centered_energy_retained_384d':energy_ratio(base[test]-bm,w,384),
            'test_paired_cosine':float(F.cosine_similarity(small[test],aligned[test]).mean()),
            'endpoint_pca_training_variance':{str(k):float(ep_eigen[:k].sum()/ep_eigen.sum()) for k in [2,3]},
            'endpoint_pca_test_displacement_energy':{str(k):energy_ratio(dt,ep_axes,k) for k in [2,3]},
            'displacement_pca_training_variance':{str(k):float(d_eigen[:k].sum()/d_eigen.sum()) for k in [2,3,10,50]},
            'displacement_pca_test_centered_energy':{str(k):energy_ratio(dt-d_mean,d_axes,k) for k in [2,3,10,50]},
            'train_residual_rms':float(delta[train].square().sum(-1).mean().sqrt()),
            'validation_residual_rms':float(delta[valid].square().sum(-1).mean().sqrt())}
    # Test arrows are compared with neighbors in the TRAINING origin geometry.
    origin_train=F.normalize(small[train],dim=-1)
    g=torch.Generator().manual_seed(2026)
    nearby=[]; random=[]; predictions=[]
    for batch in torch.tensor(test).split(128):
        neighbors=(F.normalize(small[batch],dim=-1)@origin_train.T).topk(10,dim=-1).indices
        neighbor_delta=delta[train][neighbors]
        random_delta=delta[train][torch.randint(len(train),neighbors.shape,generator=g)]
        nearby.extend(F.cosine_similarity((delta[batch]-d_mean)[:,None,:],neighbor_delta-d_mean,dim=-1).mean(-1).tolist())
        random.extend(F.cosine_similarity((delta[batch]-d_mean)[:,None,:],random_delta-d_mean,dim=-1).mean(-1).tolist())
        predictions.append(neighbor_delta.mean(1))
    pred=torch.cat(predictions)
    base_mse=float((dt-d_mean).square().sum(-1).mean())
    result['locality']={'neighbors':10,'centered_neighbor_direction_cosine':float(np.mean(nearby)),
                        'centered_random_direction_cosine':float(np.mean(random)),
                        'paired_neighbor_minus_random':{**paired_bootstrap(nearby,random),'unit':'held-out text; one fitted alignment'},
                        'knn_displacement_mse':float((dt-pred).square().sum(-1).mean()),
                        'global_mean_displacement_mse':base_mse,
                        'knn_mse_improvement_over_mean':1-float((dt-pred).square().sum(-1).mean())/base_mse}
    projected={}
    for name,mean,axes in [('endpoints',ep_mean,ep_axes),('displacements',ep_mean,d_axes)]:
        projected[name]={'origins':((small[test]-mean)@axes[:,:3]).tolist(),
                         'ends':((aligned[test]-mean)@axes[:,:3]).tolist()}
    return result,projected,delta,ep_eigen,d_eigen


def retrieval(data,queries=None,documents=None):
    queries=data['queries'] if queries is None else queries
    documents=data['documents'] if documents is None else documents
    queries=F.normalize(queries,dim=-1);documents=F.normalize(documents,dim=-1)
    rows=[]
    for i in data['splits']['test']:
        rank=(queries[i]@documents.T).argsort(descending=True,stable=True)[:10].tolist()
        rows.append({'query_id':data['query_ids'][i],**metrics(rank,data['qrels'][i])})
    return {key:float(np.mean([r[key] for r in rows])) for key in ['ndcg@10','recall@10','mrr@10']},rows


def run(small,base,output,cache):
    started=time.perf_counter();torch.set_num_threads(4)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for key in ['query_ids','document_ids','splits','official_candidates','qrels']:
        assert small[key]==base[key],f'Unmatched data: {key}'
    assert small['metadata']['source_sha256']==base['metadata']['source_sha256']
    qsplit=official_splits(small,cache)
    dsets={s:set(d for q in qs for d in small['official_candidates'][q]) for s,qs in qsplit.items()}
    # Shared sentence text is removed from field fitting/evaluation splits.
    exclusive={s:sorted(ds-set().union(*(dsets[t] for t in dsets if t!=s))) for s,ds in dsets.items()}
    n=len(small['queries'])
    x=torch.cat([small['queries'],small['documents']]).double()
    y=torch.cat([base['queries'],base['documents']]).double()
    split={s:qsplit[s]+[n+i for i in exclusive[s]] for s in qsplit}
    w,bm,sm=align(y[split['train']],x[split['train']])
    stats,projection,delta,ee,de=sample_statistics(x,y,split,w,bm,sm)
    groupstats={}
    for kind, subset in [('questions',qsplit['test']),('answers',[n+i for i in exclusive['test']])]:
        d=delta[subset]
        groupstats[kind]={'count':len(subset),'mean_displacement_norm':float(d.norm(dim=-1).mean()),
                         'rms_displacement':float(d.square().sum(-1).mean().sqrt())}
    # Within-type locality avoids mistaking question/answer separation for smoothness.
    for kind, train_ids, test_ids in [('questions',qsplit['train'],qsplit['test']),('answers',[n+i for i in exclusive['train']],[n+i for i in exclusive['test']])]:
        center=delta[train_ids].mean(0); near=[]; rand=[]; pred=[]
        g=torch.Generator().manual_seed(2026)
        for ids_batch in torch.tensor(test_ids).split(128):
            neighbors=(F.normalize(x[ids_batch],dim=-1)@F.normalize(x[train_ids],dim=-1).T).topk(10,dim=-1).indices
            dn=delta[train_ids][neighbors]
            dr=delta[train_ids][torch.randint(len(train_ids),neighbors.shape,generator=g)]
            near.extend(F.cosine_similarity((delta[ids_batch]-center)[:,None,:],dn-center,dim=-1).mean(-1).tolist())
            rand.extend(F.cosine_similarity((delta[ids_batch]-center)[:,None,:],dr-center,dim=-1).mean(-1).tolist())
            pred.append(dn.mean(1))
        baseline=(delta[test_ids]-center).square().sum(-1).mean()
        groupstats[kind]['locality']={'neighbor_direction_cosine':float(np.mean(near)),'random_direction_cosine':float(np.mean(rand)),
            'knn_mse_improvement_over_mean':1-float((delta[test_ids]-torch.cat(pred)).square().sum(-1).mean()/baseline)}
    g=torch.Generator().manual_seed(2026)
    train_ids=split['train'];permuted=torch.randperm(len(train_ids),generator=g)
    null_w,null_bm,null_sm=align(y[train_ids][permuted],x[train_ids])
    stats['shuffled_alignment_test_rms']=float((((y[split['test']]-null_bm)@null_w+null_sm-x[split['test']]).square().sum(-1).mean()).sqrt())
    native={};native_rows={}
    for name,data in [('small',small),('base',base)]:
        native[name],native_rows[name]=retrieval(data)
    native['aligned_base'],native_rows['aligned_base']=retrieval(base,((base['queries'].double()-bm)@w+sm).float(),((base['documents'].double()-bm)@w+sm).float())
    comparison={**paired_bootstrap([r['ndcg@10'] for r in native_rows['base']],[r['ndcg@10'] for r in native_rows['small']]),'unit':'held-out answerable question; fixed pretrained encoders'}
    # Zero-padded small embeddings lifted by W.T preserve their pairwise distances;
    # this measures base residual energy excluded by the compressed primary view.
    centered_base=y[split['test']]-bm
    discarded=centered_base-(centered_base@w)@w.T
    full_residual=centered_base-(x[split['test']]-sm)@w.T
    stats['compressed_field_fraction_of_full_lifted_residual_energy']=float(delta[split['test']].square().sum()/full_residual.square().sum())
    stats['discarded_base_residual_energy_fraction']=float(discarded.square().sum()/full_residual.square().sum())
    rng=np.random.default_rng(2026)
    test=split['test'];sample=sorted(rng.choice(len(test),min(500,len(test)),replace=False).tolist())
    ids=small['query_ids']+small['document_ids']
    points=[{'id':ids[test[j]],'kind':'question' if test[j]<n else 'answer',
             'norm':float(delta[test[j]].norm()),
             **{name:{'start':v['origins'][j],'end':v['ends'][j]} for name,v in projection.items()}} for j in sample]
    (out/'points.json').write_text(json.dumps(points)+'\n')
    spectra={'endpoint_variance_ratio':(ee/ee.sum()).tolist(),'displacement_variance_ratio':(de/de.sum()).tolist()}
    payload={'statistics':stats,'groups':groupstats,'retrieval':native,'base_minus_small_ndcg':comparison,
             'split_counts':{s:{'questions':len(qsplit[s]),'exclusive_answers':len(exclusive[s]),'total':len(split[s])} for s in split},
             'excluded_shared_answers':len(set().union(*dsets.values()))-sum(map(len,exclusive.values())),
             'provenance':{'small':small['metadata'],'base':base['metadata']},'spectra':spectra,
             'analysis_seconds':time.perf_counter()-started,'sample_seed':2026,'display_points':len(points),
             'alignment':'centered rectangular orthogonal Procrustes base 768 -> small 384; no fitted scale; train texts only',
             'pca':'pooled train endpoints for spatial view; centered train displacement PCA for movement view',
             'hashes':{name:hashlib.sha256(torch.cat([d['queries'],d['documents']]).numpy().tobytes()).hexdigest() for name,d in [('small',small),('base',base)]}}
    (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    (out/'retrieval-per-query.json').write_text(json.dumps(native_rows,indent=2)+'\n')
    (out/'splits.json').write_text(json.dumps({s:[ids[i] for i in indexes] for s,indexes in split.items()},indent=2)+'\n')
    torch.save({'base_to_small':w,'base_mean':bm,'small_mean':sm},out/'alignment.pt')
    print(json.dumps({k:v for k,v in payload.items() if k not in ['provenance','spectra','hashes']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--small',default='cache/wikiqa-bge.pt');p.add_argument('--base',default='cache/wikiqa-bge-base.pt');p.add_argument('--output',required=True);p.add_argument('--hf-cache',default='cache/huggingface')
    a=p.parse_args();run(torch.load(a.small,weights_only=True),torch.load(a.base,weights_only=True),a.output,a.hf_cache)
