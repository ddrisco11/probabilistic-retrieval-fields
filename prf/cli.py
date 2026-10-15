import argparse
import json
from pathlib import Path
import torch
from .data import synthetic, encode_beir
from .experiment import run


def main():
    p=argparse.ArgumentParser(description='PRF reproducible retrieval experiments')
    sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run')
    r.add_argument('--config',default='configs/synthetic.json')
    r.add_argument('--data',help='Local trusted embedding bundle, produced by encode')
    r.add_argument('--output',required=True)
    e=sub.add_parser('encode')
    e.add_argument('--folder',required=True)
    e.add_argument('--output',required=True)
    e.add_argument('--model',default='BAAI/bge-small-en-v1.5')
    e.add_argument('--revision',default='main',help='Resolved immutable commit is saved in provenance')
    args=p.parse_args()
    if args.command=='encode':
        encode_beir(Path(args.folder),args.output,args.model,args.revision)
    else:
        config=json.loads(Path(args.config).read_text())
        data=torch.load(args.data,weights_only=True) if args.data else synthetic(**config.get('synthetic',{}))
        run(data,config,args.output)

if __name__=='__main__':
    main()
