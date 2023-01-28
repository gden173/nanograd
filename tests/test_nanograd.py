import torch
import pytest
from nanograd.nanograd import Val

"""Copies the tests from micrograd 
   to ensure consistency"""

TOLERANCE = 1e-6


def test_addition():
    na = Val(-4.0)
    nb = Val(8.0)
    nc = na + nb
    nc.backward()
    at = torch.tensor(-4.0, requires_grad=True)
    bt = torch.tensor(8.0, requires_grad=True)
    ct = at + bt
    ct.retain_grad()
    ct.backward()
    assert nc.item() == ct.item()
    assert at.grad.item() == na.grad
    assert bt.grad.item() == nb.grad
    assert ct.grad.item() == nc.grad


def test_multiplication():
    na = Val(-4.0)
    nb = Val(8.0)
    nc = na * nb
    nc.backward()
    at = torch.tensor(-4.0, requires_grad=True)
    bt = torch.tensor(8.0, requires_grad=True)
    ct = at * bt
    ct.retain_grad()
    ct.backward()

    assert nc.item() == ct.item()
    assert at.grad.item() == na.grad
    assert bt.grad.item() == nb.grad
    assert ct.grad.item() == nc.grad


def test_division():
    na = Val(-4.0)
    nb = Val(8.0)
    nc = na / nb
    nc.backward()
    at = torch.tensor(-4.0, requires_grad=True)
    bt = torch.tensor(8.0, requires_grad=True)
    ct = at / bt
    ct.retain_grad()
    ct.backward()

    assert abs(nc.item() - ct.item()) < TOLERANCE
    assert abs(at.grad.item() - na.grad) < TOLERANCE
    assert abs(bt.grad.item() - nb.grad) < TOLERANCE
    assert abs(ct.grad.item() - nc.grad) < TOLERANCE


def test_powers():
    na = Val(4.0)
    nc = na**4.5
    nc.backward()
    at = torch.tensor(4.0, requires_grad=True)
    ct = at**4.5
    ct.retain_grad()
    ct.backward()

    assert abs(nc.item() - ct.item()) < TOLERANCE
    assert abs(at.grad.item() - na.grad) < TOLERANCE
    assert abs(ct.grad.item() - nc.grad) < TOLERANCE


def test_exp():
    na = Val(2.0)
    nc = na.exp()
    nc.backward()
    at = torch.tensor(2.0, requires_grad=True)
    ct = at.exp()
    ct.retain_grad()
    ct.backward()

    assert abs(nc.item() - ct.item()) < TOLERANCE
    assert abs(at.grad.item() - na.grad) < TOLERANCE
    assert abs(ct.grad.item() - nc.grad) < TOLERANCE


def test_tanh():
    na = Val(2.0)
    nc = na.tanh()
    nc.backward()
    print(na.grad)

    at = torch.tensor(2.0, requires_grad=True)
    ct = at.tanh()
    ct.retain_grad()
    ct.backward()
    assert abs(nc.item() - ct.item()) < TOLERANCE
    assert abs(at.grad.item() - na.grad) < TOLERANCE
    assert abs(ct.grad.item() - nc.grad) < TOLERANCE


def test_sigmoid():
    na = Val(2.0)
    nc = na.sigmoid()
    nc.backward()
    print(na.grad)

    at = torch.tensor(2.0, requires_grad=True)
    ct = at.sigmoid()
    ct.retain_grad()
    ct.backward()
    assert abs(nc.item() - ct.item()) < TOLERANCE
    assert abs(at.grad.item() - na.grad) < TOLERANCE
    assert abs(ct.grad.item() - nc.grad) < TOLERANCE


"""Micrograds Test"""


def test_sanity_check_relu():
    x = Val(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y
    assert ymg.data == ypt.data.item()
    assert xmg.grad == xpt.grad.item()


def test_sanity_check_tanh():
    x = Val(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y
    assert ymg.data == ypt.data.item()
    assert xmg.grad == xpt.grad.item()


def test_sanity_check_sigmoid():
    x = Val(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).sigmoid()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).sigmoid()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y
    assert ymg.data == ypt.data.item()
    assert xmg.grad == xpt.grad.item()


def test_more_ops():
    a = Val(-4.0)
    b = Val(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    assert abs(gmg.data - gpt.data.item()) < tol
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
