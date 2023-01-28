"""Copies the tests from micrograd 
   to ensure consistenanograd_c_valy"""
# pylint: skip-file
from torch import tensor
from nanograd.nanograd import Val

TOLERANCE = 1e-6


def test_addition():
    """Test addition"""
    nanograd_a_val = Val(-4.0)
    nanograd_b_val = Val(8.0)
    nanograd_c_val = nanograd_a_val + nanograd_b_val
    nanograd_c_val.backward()
    torch_a_val = tensor(-4.0, requires_grad=True)
    torch_b_val = tensor(8.0, requires_grad=True)
    torch_c_val = torch_a_val + torch_b_val
    torch_c_val.retain_grad()
    torch_c_val.backward()
    assert nanograd_c_val.item() == torch_c_val.item()
    assert torch_a_val.grad.item() == nanograd_a_val.grad
    assert torch_b_val.grad.item() == nanograd_b_val.grad
    assert torch_c_val.grad.item() == nanograd_c_val.grad


def test_multiplication():
    """Test multiplication"""
    nanograd_a_val = Val(-4.0)
    nanograd_b_val = Val(8.0)
    nanograd_c_val = nanograd_a_val * nanograd_b_val
    nanograd_c_val.backward()
    torch_a_val = tensor(-4.0, requires_grad=True)
    torch_b_val = tensor(8.0, requires_grad=True)
    torch_c_val = torch_a_val * torch_b_val
    torch_c_val.retain_grad()
    torch_c_val.backward()

    assert nanograd_c_val.item() == torch_c_val.item()
    assert torch_a_val.grad.item() == nanograd_a_val.grad
    assert torch_b_val.grad.item() == nanograd_b_val.grad
    assert torch_c_val.grad.item() == nanograd_c_val.grad


def test_division():
    """Test division"""
    nanograd_a_val = Val(-4.0)
    nanograd_b_val = Val(8.0)
    nanograd_c_val = nanograd_a_val / nanograd_b_val
    nanograd_c_val.backward()
    torch_a_val = tensor(-4.0, requires_grad=True)
    torch_b_val = tensor(8.0, requires_grad=True)
    torch_c_val = torch_a_val / torch_b_val
    torch_c_val.retain_grad()
    torch_c_val.backward()

    assert abs(nanograd_c_val.item() - torch_c_val.item()) < TOLERANCE
    assert abs(torch_a_val.grad.item() - nanograd_a_val.grad) < TOLERANCE
    assert abs(torch_b_val.grad.item() - nanograd_b_val.grad) < TOLERANCE
    assert abs(torch_c_val.grad.item() - nanograd_c_val.grad) < TOLERANCE


def test_powers():
    """Test powers"""
    nanograd_a_val = Val(4.0)
    nanograd_c_val = nanograd_a_val**4.5
    nanograd_c_val.backward()
    torch_a_val = tensor(4.0, requires_grad=True)
    torch_c_val = torch_a_val**4.5
    torch_c_val.retain_grad()
    torch_c_val.backward()

    assert abs(nanograd_c_val.item() - torch_c_val.item()) < TOLERANCE
    assert abs(torch_a_val.grad.item() - nanograd_a_val.grad) < TOLERANCE
    assert abs(torch_c_val.grad.item() - nanograd_c_val.grad) < TOLERANCE


def test_exp():
    """Test Exp"""
    nanograd_a_val = Val(2.0)
    nanograd_c_val = nanograd_a_val.exp()
    nanograd_c_val.backward()
    torch_a_val = tensor(2.0, requires_grad=True)
    torch_c_val = torch_a_val.exp()
    torch_c_val.retain_grad()
    torch_c_val.backward()

    assert abs(nanograd_c_val.item() - torch_c_val.item()) < TOLERANCE
    assert abs(torch_a_val.grad.item() - nanograd_a_val.grad) < TOLERANCE
    assert abs(torch_c_val.grad.item() - nanograd_c_val.grad) < TOLERANCE


def test_tanh():
    """Test tanh"""
    nanograd_a_val = Val(2.0)
    nanograd_c_val = nanograd_a_val.tanh()
    nanograd_c_val.backward()
    print(nanograd_a_val.grad)

    torch_a_val = tensor(2.0, requires_grad=True)
    torch_c_val = torch_a_val.tanh()
    torch_c_val.retain_grad()
    torch_c_val.backward()
    assert abs(nanograd_c_val.item() - torch_c_val.item()) < TOLERANCE
    assert abs(torch_a_val.grad.item() - nanograd_a_val.grad) < TOLERANCE
    assert abs(torch_c_val.grad.item() - nanograd_c_val.grad) < TOLERANCE


def test_sigmoid():
    """Test sigmoid"""
    nanograd_a_val = Val(2.0)
    nanograd_c_val = nanograd_a_val.sigmoid()
    nanograd_c_val.backward()
    print(nanograd_a_val.grad)

    torch_a_val = tensor(2.0, requires_grad=True)
    torch_c_val = torch_a_val.sigmoid()
    torch_c_val.retain_grad()
    torch_c_val.backward()
    assert abs(nanograd_c_val.item() - torch_c_val.item()) < TOLERANCE
    assert abs(torch_a_val.grad.item() - nanograd_a_val.grad) < TOLERANCE
    assert abs(torch_c_val.grad.item() - nanograd_c_val.grad) < TOLERANCE


# Micrograds Test
def test_relu():
    """Test relu"""
    x_var = Val(-4.0)
    z_var = 2 * x_var + 2 + x_var
    q_var = z_var.relu() + z_var * x_var
    h_var = (z_var * z_var).relu()
    y_var = h_var + q_var + q_var * x_var
    y_var.backward()
    x_grad_var, y_grad_var = x_var, y_var

    x_var = tensor([-4.0]).double()
    x_var.requires_grad = True
    z_var = 2 * x_var + 2 + x_var
    q_var = z_var.relu() + z_var * x_var
    h_var = (z_var * z_var).relu()
    y_var = h_var + q_var + q_var * x_var
    y_var.backward()
    x_torch_var, y_torch_var = x_var, y_var
    assert y_grad_var.data == y_torch_var.data.item()
    assert x_grad_var.grad == x_torch_var.grad.item()


def test_tanh_2():
    """Test tanh additional"""
    x_var = Val(-4.0)
    z_var = 2 * x_var + 2 + x_var
    q_var = z_var.relu() + z_var * x_var
    h_var = (z_var * z_var).tanh()
    y_var = h_var + q_var + q_var * x_var
    y_var.backward()
    x_nano_grad, y_nano_grad = x_var, y_var

    x_var = tensor([-4.0]).double()
    x_var.requires_grad = True
    z_var = 2 * x_var + 2 + x_var
    q_var = z_var.relu() + z_var * x_var
    h_var = (z_var * z_var).tanh()
    y_var = h_var + q_var + q_var * x_var
    y_var.backward()
    x_torch_grad, y_torch_grad = x_var, y_var
    assert y_nano_grad.data == y_torch_grad.data.item()
    assert x_nano_grad.grad == x_torch_grad.grad.item()


def test_sigmoid_2():
    """Test sigmoid additional"""
    x_var = Val(-4.0)
    z_var = 2 * x_var + 2 + x_var
    q_var = z_var.relu() + z_var * x_var
    h_var = (z_var * z_var).sigmoid()
    y_var = h_var + q_var + q_var * x_var
    y_var.backward()
    x_nano_grad, y_nano_grad = x_var, y_var

    x_var = tensor([-4.0]).double()
    x_var.requires_grad = True
    z_var = 2 * x_var + 2 + x_var
    q_var = z_var.relu() + z_var * x_var
    h_var = (z_var * z_var).sigmoid()
    y_var = h_var + q_var + q_var * x_var
    y_var.backward()
    x_torch_grad, y_torch_grad = x_var, y_var
    assert y_nano_grad.data == y_torch_grad.data.item()
    assert x_nano_grad.grad == x_torch_grad.grad.item()


def test_more_ops():
    """Test extended operations"""
    a_var = Val(-4.0)
    b_var = Val(2.0)
    c_var = a_var + b_var
    d_var = a_var * b_var + b_var**3
    c_var += c_var + 1
    c_var += 1 + c_var + (-a_var)
    d_var += d_var * 2 + (b_var + a_var).relu()
    d_var += 3 * d_var + (b_var - a_var).relu()
    e_var = c_var - d_var
    f_var = e_var**2
    g_var = f_var / 2.0
    g_var += 10.0 / f_var
    g_var.backward()
    a_nano_grad, b_nano_grad, c_nano_grad = a_var, b_var, g_var

    a_var = tensor([-4.0]).double()
    b_var = tensor([2.0]).double()
    a_var.requires_grad = True
    b_var.requires_grad = True
    c_var = a_var + b_var
    d_var = a_var * b_var + b_var**3
    c_var = c_var + c_var + 1
    c_var = c_var + 1 + c_var + (-a_var)
    d_var = d_var + d_var * 2 + (b_var + a_var).relu()
    d_var = d_var + 3 * d_var + (b_var - a_var).relu()
    e_var = c_var - d_var
    f_var = e_var**2
    g_var = f_var / 2.0
    g_var = g_var + 10.0 / f_var
    g_var.backward()
    a_torch, b_torch, g_torch = a_var, b_var, g_var

    tol = 1e-6
    assert abs(c_nano_grad.data - g_torch.data.item()) < tol
    assert abs(a_nano_grad.grad - a_torch.grad.item()) < tol
    assert abs(b_nano_grad.grad - b_torch.grad.item()) < tol
