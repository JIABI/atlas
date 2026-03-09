
from .primitives import x0_target, u_target, r_target, eps_target

def line_x0_u(alpha, x0, xt, eps, t):
    return alpha*x0 + (1-alpha)*u_target(x0,xt,eps,t)

def line_x0_r(alpha, x0, xt, eps, t):
    return alpha*x0 + (1-alpha)*r_target(x0,xt,eps,t)

def line_x0_eps(alpha, x0, xt, eps, t):
    return alpha*x0 + (1-alpha)*eps_target(x0,xt,eps,t)
