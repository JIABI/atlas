
import torch

def x0_target(x0, xt, eps, t): return x0

def u_target(x0, xt, eps, t): return x0-xt

def r_target(x0, xt, eps, t): return xt-eps

def eps_target(x0, xt, eps, t): return eps
