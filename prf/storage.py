"""Compact numeric result storage; raw source text is never included."""
import gzip
import json
import numpy as np


def save_region_results(payload,out):
    summaries=[]
    # mtime=0 makes the compression header reproducible.
    with (out/'per-query.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:
            for result in payload['results']:
                identity={k:result[k] for k in ['method','regions','probes','seed']}
                for row in result['per_query']:
                    f.write((json.dumps({**identity,**row})+'\n').encode())
                summary={k:v for k,v in result.items() if k!='per_query'}
                summary['p50_ms']=float(np.median([r['total_ms'] for r in result['per_query']]))
                summary['p95_ms']=float(np.quantile([r['total_ms'] for r in result['per_query']],.95))
                summary['per_query_file']='per-query.jsonl.gz'
                summaries.append(summary)
    compact=dict(payload,results=summaries)
    (out/'results.json').write_text(json.dumps(compact,indent=2)+'\n')
