"""Frozen BGE encoding of WikiQA answer sentences; official question splits."""
import hashlib
import json
import time
from pathlib import Path
import torch

QUERY_PREFIX='Represent this sentence for searching relevant passages: '


def prepare(output, dataset_revision='main', model_revision='main', model_name='BAAI/bge-small-en-v1.5'):
    from huggingface_hub import HfApi, hf_hub_download
    from sentence_transformers import SentenceTransformer
    import pyarrow.parquet as pq
    torch.set_num_threads(4)
    api=HfApi()
    dataset='microsoft/wiki_qa'
    ds_sha=api.dataset_info(dataset,revision=dataset_revision).sha
    model_sha=api.model_info(model_name,revision=model_revision).sha
    files=api.list_repo_files(dataset,repo_type='dataset',revision=ds_sha)
    rows={};hashes={}
    for split in ['train','validation','test']:
        matching=[f for f in files if f.endswith('.parquet') and Path(f).name.startswith(split+'-')]
        if not matching:raise ValueError(f'No parquet files for {split}')
        rows[split]=[]
        for f in sorted(matching):
            path=hf_hub_download(dataset,f,repo_type='dataset',revision=ds_sha)
            hashes[f]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            rows[split].extend(pq.read_table(path).to_pylist())
    docs=sorted({r['answer'].strip() for split in rows.values() for r in split})
    docmap={s:i for i,s in enumerate(docs)}
    questions={};qrels={};split_ids={};candidates={}
    for split,records in rows.items():
        ids=set()
        for r in records:
            qid=r['question_id'];ids.add(qid)
            if qid in questions and questions[qid]!=r['question']:
                raise ValueError('Inconsistent question text')
            questions[qid]=r['question']
            candidates.setdefault(qid,set()).add(docmap[r['answer'].strip()])
            qrels.setdefault(qid,{})
            if r['label']==1:qrels[qid][docmap[r['answer'].strip()]]=1
        split_ids[split]=ids
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if split_ids[a]&split_ids[b]:raise ValueError('Question ID leakage')
        if {questions[q] for q in split_ids[a]}&{questions[q] for q in split_ids[b]}:
            raise ValueError('Identical question text across official splits')
    qids=sorted(questions)
    qmap={q:i for i,q in enumerate(qids)}
    splits={s:sorted(qmap[q] for q in ids if qrels[q]) for s,ids in split_ids.items()}
    excluded={s:sum(not qrels[q] for q in ids) for s,ids in split_ids.items()}
    started=time.perf_counter()
    encoder=SentenceTransformer(model_name,revision=model_sha,device='cpu')
    encoder.eval()
    for p in encoder.parameters():p.requires_grad_(False)
    d=encoder.encode(docs,normalize_embeddings=True,convert_to_tensor=True,batch_size=64,show_progress_bar=True)
    q=encoder.encode([QUERY_PREFIX+questions[i] for i in qids],normalize_embeddings=True,
                     convert_to_tensor=True,batch_size=64,show_progress_bar=True)
    metadata={'kind':'wikiqa','dataset':dataset,'dataset_revision':ds_sha,'source_sha256':hashes,
              'model':model_name,'revision':model_sha,'query_prefix':QUERY_PREFIX,'document_prefix':'',
              'normalize':True,'dimension':d.shape[1],'max_seq_length':encoder.max_seq_length,
              'encoding_seconds':time.perf_counter()-started,'device':'cpu','encoding_threads':4,
              'corpus_sentences':len(docs),'split_answerable_queries':{s:len(i) for s,i in splits.items()},
              'excluded_no_positive_queries':excluded,'source_pair_counts':{s:len(r) for s,r in rows.items()},
              'protocol':'Global deduplicated answer corpus; not official candidate-restricted WikiQA leaderboard.'}
    bundle={'documents':d.cpu(),'queries':q.cpu(),'qrels':[qrels[i] for i in qids],
            'splits':splits,'types':['wikiqa']*len(qids),'groups':None,'query_ids':qids,
            'document_ids':[hashlib.sha256(s.encode()).hexdigest() for s in docs],
            'official_candidates':[sorted(candidates[i]) for i in qids],'metadata':metadata}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(bundle,out);Path(str(out)+'.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='cache/wikiqa-bge.pt')
    p.add_argument('--dataset-revision',default='main');p.add_argument('--model-revision',default='main')
    p.add_argument('--model-name',default='BAAI/bge-small-en-v1.5')
    a=p.parse_args();prepare(a.output,a.dataset_revision,a.model_revision,a.model_name)
