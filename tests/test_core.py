import math
import pytest
import torch
from torch.nn import functional as F
from prf.models import Policy
from prf.retrieval import metrics, retrieve, paired_bootstrap
from prf.data import synthetic, read_beir
from prf.experiment import relevance_surrogate, reward_loss, evaluate


def test_metrics_hand_calculated():
    got=metrics([2,0,1],{0:2,1:1},3)
    expected=(3/math.log2(3)+1/math.log2(4))/(3+1/math.log2(3))
    assert got['ndcg@3']==pytest.approx(expected)
    assert got['recall@3']==1
    assert got['mrr@3']==.5
    assert metrics([0,0],{0:1,1:1},10)['recall@10']==.5


def test_exact_budget_ties_and_deduplication():
    docs=F.normalize(torch.tensor([[1.,0.],[1.,0.],[0.,1.],[-1.,0.]]),dim=-1)
    rank,union,cost=retrieve(torch.tensor([[1.,0.],[0.,1.]]),docs,5)
    assert cost['returned_slots']==5
    assert cost['scored_pairs']==8
    assert rank==[0,1,2]
    assert len(union)==len(set(union))
    with pytest.raises(ValueError):retrieve(docs,docs,2)


@pytest.mark.parametrize('kind',['deterministic','multi','gaussian','mixture'])
def test_surrogate_gradients(kind):
    torch.manual_seed(7)
    p=Policy(9,kind,16,3)
    q=F.normalize(torch.randn(3,9),dim=-1)
    docs=F.normalize(torch.randn(8,9),dim=-1)
    target=torch.zeros(3,8);target[:,0]=1
    loss=relevance_surrogate(p,q,docs,target,3,.1)+.001*p.regularizer(q)
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters())
    if kind=='mixture':assert p.weight.weight.grad.abs().sum()>0


def test_score_function_mean_gradient():
    torch.manual_seed(9)
    mu=torch.tensor(0.,requires_grad=True)
    dist=torch.distributions.Normal(mu,1.)
    action=dist.sample((100000,)).detach()
    (action*dist.log_prob(action)).mean().backward()
    assert mu.grad.item()==pytest.approx(1,abs=.02)


def test_reward_gradients_and_bounds():
    torch.manual_seed(3)
    p=Policy(8,'mixture',16,2)
    q=F.normalize(torch.randn(4,8),dim=-1)
    docs=F.normalize(torch.randn(20,8),dim=-1)
    pos=torch.zeros(4,20);pos[:,0]=1
    loss,reward=reward_loss(p,q,docs,pos,8,3)
    loss.backward()
    assert reward.min()>=0 and reward.max()<=1
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters())
    assert p.mean.weight.grad.abs().sum()>0


def test_split_reproducibility_and_no_rng_consumption():
    d=synthetic(queries=60)
    assert d['splits']==synthetic(queries=60)['splits']
    sets=[set(x) for x in d['splits'].values()]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    p=Policy(24,'gaussian')
    state=torch.random.get_rng_state()
    a,_=evaluate(p,d,d['splits']['test'],4,40)
    assert torch.equal(state,torch.random.get_rng_state())
    b,_=evaluate(p,d,d['splits']['test'],4,40)
    assert a['ndcg@10']==b['ndcg@10']


def test_bootstrap_identical_pairs():
    assert paired_bootstrap([.1,.5,1],[.1,.5,1])['ci95']==[0.,0.]


def test_kl_zero_at_prior():
    p=Policy(8,'mixture',16,3)
    with torch.no_grad():p.mean.weight.zero_();p.mean.bias.zero_()
    q=F.normalize(torch.randn(4,8),dim=-1)
    assert abs(float(p.regularizer(q).detach()))<1e-5


def test_beir_leakage_rejected(tmp_path):
    import json
    (tmp_path/'qrels').mkdir()
    (tmp_path/'queries.jsonl').write_text(json.dumps({'_id':'q','text':'query'})+'\n')
    (tmp_path/'corpus.jsonl').write_text(json.dumps({'_id':'d','text':'doc'})+'\n')
    for split in ['train','test']:
        (tmp_path/'qrels'/f'{split}.tsv').write_text('query-id\tcorpus-id\tscore\nq\td\t1\n')
    with pytest.raises(ValueError,match='leakage'):read_beir(tmp_path)


def test_region_partition_and_multi_positive_targets():
    from prf.regions import partition,region_targets,route_metrics,select_regions
    docs=F.normalize(torch.tensor([[1.,.1],[1.,-.1],[-1.,.1],[-1.,-.1]]),dim=-1)
    centers,labels=partition(docs,2)
    assert labels[0]==labels[1] and labels[2]==labels[3] and labels[0]!=labels[2]
    targets=region_targets([{0:1,2:1},{0:1}],labels,2)
    assert torch.allclose(targets[0],torch.tensor([.5,.5]))
    p=torch.tensor([.9,.1]);s=select_regions(p,2,'sample',torch.Generator().manual_seed(7))
    assert len(s.unique())==2
    assert route_metrics(p,targets[0],s)['answer_region_coverage']==1
    assert route_metrics(p,targets[0],s[:1])['answer_region_coverage']==.5


def test_router_learning_reduces_train_cross_entropy():
    from prf.regions import fit_router,partition,region_targets
    d=synthetic(queries=60)
    c,l=partition(d['documents'],8)
    t=region_targets(d['qrels'],l,8)
    _,h,_=fit_router(d,c,t,dict(hidden=16,lr=.01,weight_decay=0,epochs=12,batch_size=16),11)
    assert h[-1]['train_cross_entropy']<h[0]['train_cross_entropy']


def test_searching_all_regions_matches_full_corpus():
    from prf.regions import partition,region_targets,RegionRouter,evaluate_router
    torch.set_num_threads(1)
    d=synthetic(queries=60)
    c,l=partition(d['documents'],4)
    target=region_targets(d['qrels'],l,4)
    _,regional=evaluate_router(d,c,l,target,RegionRouter(c),'centroid',4,11)
    _,dense=evaluate(Policy(24,'dense'),d,d['splits']['test'],1,len(d['documents']))
    for a,b in zip(regional,dense):
        assert a['region_hit']==1 and a['answer_region_coverage']==pytest.approx(1)
        assert a['corpus_fraction']==1
        assert a['ranking']==b['ranking']
        assert a['ndcg@10']==b['ndcg@10']


def test_compressed_measurements_roundtrip(tmp_path):
    import gzip,json
    from prf.storage import save_region_results
    original={'query_id':'q','total_ms':1.2,'region_hit':1.0,'ranking':['d']}
    payload={'results':[dict(method='centroid',regions=4,probes=1,seed=11,per_query=[original])]}
    save_region_results(payload,tmp_path)
    with gzip.open(tmp_path/'per-query.jsonl.gz','rt') as f:restored=json.loads(f.readline())
    for key,value in original.items():assert restored[key]==value
    summary=json.loads((tmp_path/'results.json').read_text())
    assert summary['results'][0]['p95_ms']==1.2
    assert 'per_query' not in summary['results'][0]
