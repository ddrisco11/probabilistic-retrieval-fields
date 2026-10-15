import torch
from prf.vector_field import align,pca,energy_ratio


def test_alignment_recovers_isometric_lift_and_translation():
    torch.manual_seed(42)
    x=torch.randn(100,3,dtype=torch.float64)
    lift=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64)).Q
    y=x@lift.T+torch.randn(7,dtype=torch.float64)
    w,bm,sm=align(y[:70],x[:70])
    torch.testing.assert_close((y[70:]-bm)@w+sm,x[70:])
    torch.testing.assert_close(w.T@w,torch.eye(3,dtype=torch.float64))


def test_same_pca_axes_preserve_arrow_projection():
    torch.manual_seed(5)
    x=torch.randn(50,5,dtype=torch.float64);y=x+torch.randn_like(x)*.1
    mean,axes,values=pca(torch.cat([x,y]))
    torch.testing.assert_close((y-mean)@axes[:,:3]-(x-mean)@axes[:,:3],(y-x)@axes[:,:3])
    assert 0<=energy_ratio(y-x,axes,2)<=energy_ratio(y-x,axes,3)<=1
    assert torch.all(values[:-1]>=values[1:])


def test_compressed_and_discarded_field_energy_decompose():
    torch.manual_seed(1)
    w=torch.linalg.qr(torch.randn(8,3,dtype=torch.float64)).Q
    x=torch.randn(20,3,dtype=torch.float64);y=torch.randn(20,8,dtype=torch.float64)
    residual=y-x@w.T
    compressed=y@w-x;discarded=y-(y@w)@w.T
    torch.testing.assert_close(residual.square().sum(),compressed.square().sum()+discarded.square().sum())
