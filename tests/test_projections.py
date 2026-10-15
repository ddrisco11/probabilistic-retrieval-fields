import torch
from prf.projections import expected_distance, expected_scores, sparsemax, WeightedProjections


def test_literal_sixty_forty_reward_and_all_heads_receive_gradients():
    points = torch.tensor([[[1.,0.],[0.,1.]]], requires_grad=True)
    weights = torch.tensor([[.6,.4]], requires_grad=True)
    answer = torch.tensor([[[1.,0.]]])
    loss = expected_distance(points,weights,answer,torch.ones(1,1))
    torch.testing.assert_close(loss, torch.tensor([.8]))
    loss.sum().backward()
    torch.testing.assert_close(weights.grad,torch.tensor([[0.,2.]]))
    assert points.grad[0,1].abs().sum()>0


def test_expected_ranking_equals_explicit_all_head_distances():
    torch.manual_seed(5)
    points=torch.randn(3,8,5); weights=torch.randn(3,8).softmax(-1); docs=torch.randn(11,5)
    explicit=-(weights[:,:,None]*(points[:,:,None,:]-docs[None,None,:,:]).square().sum(-1)).sum(1)
    torch.testing.assert_close(expected_scores(points,weights,docs),explicit)


def test_sparse_gate_preserves_mass_and_can_choose_different_counts():
    logits=torch.tensor([[3.,0.,-2.],[0.,0.,0.]],requires_grad=True)
    weights=sparsemax(logits)
    torch.testing.assert_close(weights,torch.tensor([[1.,0.,0.],[1/3,1/3,1/3]]))
    weights[:,0].sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_multiple_answers_are_averaged_without_padding_credit():
    z=torch.tensor([[[1.,0.]]]); w=torch.ones(1,1)
    a=torch.tensor([[[1.,0.],[0.,1.],[100.,100.]]]); mask=torch.tensor([[1.,1.,0.]])
    torch.testing.assert_close(expected_distance(z,w,a,mask),torch.tensor([1.]))


def test_projection_can_move_and_preserves_unit_norm():
    model=WeightedProjections(4,8,gate='sparsemax')
    q=torch.nn.functional.normalize(torch.randn(3,4),dim=-1)
    z,w=model(q)
    torch.testing.assert_close(z.norm(dim=-1),torch.ones(3,8))
    torch.testing.assert_close(w.sum(-1),torch.ones(3))
    assert not torch.equal(z[:,0],q)
