"""Transition matrices of the HMM families, and their order-1 and order-0 approximations.

Paper section: Section 2.1 (Hidden Markov models), Section 2.3 (Order-k approximation), and Appendix C.1 (Process
definitions) with Figure 9.

Claim: In the paper's words, the families are "edge-emitting hidden Markov models (HMMs) as simple models of natural
text, chosen such that the hidden token-generating dynamics are richer than the next-token distribution (with one
contrasting case, called Mess3 (Marzen and Crutchfield, 2017), where the dynamics are exactly as rich)."

Experiment: This module runs no experiment. Every experiment script builds its HMM from these constructors through
configs/hmm_configs.py, and the order-1 and order-0 constructions are the k-HMM baselines of Sections 4 and 5.

Saved Outputs: None. This module writes no files.

How the code works: Each family has a function that returns the list of per-token transition matrices T^(x), whose
entry (i, j) is the joint probability of emitting token x and moving from hidden state i to hidden state j
(Equation 1). The order-1 functions return the transition matrices of the family's 1-HMM, a Markov chain whose hidden
state is the previous token, with the family's one-step token statistics in closed form. The order-0 functions
return the one-state HMM that emits the family's stationary token distribution. Mess3 has no order-0 constructor
because its stationary token distribution is uniform, which the scripts use directly.
"""
import numpy as np

# Mess3 (Marzen and Crutchfield, 2017): three hidden states and three tokens. At every step the process moves to each
# of the other two states with probability x, and alpha sets how faithfully the emitted token reveals the state that
# is reached (alpha = 1 makes it exact). Its emission matrix is invertible, so beliefs and NTP are linearly equivalent
# (Section 5.2).
def mess3_matrices(alpha, x):
    b = (1 - alpha) / 2
    y = 1 - 2 * x
    ay, bx, by, ax = alpha*y, b*x, b*y, alpha*x
    return [
        np.array([[ay,bx,bx],[ax,by,bx],[ax,bx,by]]),
        np.array([[by,ax,bx],[bx,ay,bx],[bx,ax,by]]),
        np.array([[by,bx,ax],[bx,by,ax],[bx,bx,ay]]),
    ]

# The 1-HMM of Mess3 (Section 2.3): a Markov chain whose hidden state is the previous token, with Mess3's one-step
# token statistics in closed form. It is the k = 1 baseline of Sections 4 and 5.
def mess3_order_one(alpha, x):
    A = 0.5*(1-2*alpha+3*alpha**2-x+6*alpha*x-9*alpha**2*x)
    B = 0.25*(1+2*alpha-3*alpha**2+x-6*alpha*x+9*alpha**2*x)
    return [np.array([[A,0,0],[B,0,0],[B,0,0]]),np.array([[0,B,0],[0,A,0],[0,B,0]]),np.array([[0,0,B],[0,0,B],[0,0,A]])]

# Arch: four hidden states and three tokens. alpha is the probability of remaining in the current state, and the
# remaining probability mass moves to the other states. Each state has its own next-token distribution, and the
# emission matrix is rank deficient (one kernel direction), so the belief carries information that the NTP does
# not (Section 6.1).
def arch_matrices(alpha):
    b = (1 - alpha) / 3
    return [
        np.array([[0.8*alpha,0,0,0],[0,0.2*alpha,0,0],[0,0,0.4*alpha,0],[0,0,0,0.6*alpha]]),
        np.array([[0,0,0,0],[0,0.4*alpha,0,0.4*b],[0,0,0.3*alpha,0],[0,0,0,0.16*alpha]]),
        np.array([[0.2*alpha,b,b,b],[b,0.4*alpha,b,0.6*b],[b,b,0.3*alpha,b],[b,b,b,0.24*alpha]]),
    ]

# The 1-HMM and the 0-HMM of Arch (Section 2.3).
def arch_order_one(alpha):
    d1=20+109*alpha
    d2=-580+409*alpha
    return [
        np.array([[3*alpha/5,0,0],[6*alpha*(10+27*alpha)/(5*d1),0,0],[18*alpha*(-80+59*alpha)/(5*d2),0,0]]),
        np.array([[0,(10+101*alpha)/750,0],[0,alpha*(560+1507*alpha)/(50*d1),0],[0,(-1000-4690*alpha+3527*alpha**2)/(50*d2),0]]),
        np.array([[0,0,(740-551*alpha)/750],[0,0,(1000+4290*alpha-3127*alpha**2)/(50*d1)],[0,0,-(28000-39540*alpha+14147*alpha**2)/(50*d2)]]),
    ]

def arch_order_zero(alpha):
    p1=alpha/2
    p2=(20+109*alpha)/600
    return [np.array([[p1]]),np.array([[p2]]),np.array([[1-p1-p2]])]

# Wing: three hidden states and two tokens. alpha is the probability of remaining in the current state. While the
# state persists, the two outer states emit token 1 and the middle state emits token 0 with probability x; a change of
# state emits token 0 or token 1 depending on the pair of states.
def wing_matrices(alpha, x):
    b = (1 - alpha) / 2
    return [
        np.array([[0,b,0],[0,x*alpha,0.5*b],[b,0,0]]),
        np.array([[alpha,0,b],[b,(1-x)*alpha,0.5*b],[0,b,alpha]]),
    ]

# The 1-HMM and the 0-HMM of Wing (Section 2.3).
def wing_order_one(alpha, x):
    p=2-4*alpha+2*alpha**2+3*alpha*x-3*alpha**2*x+4*alpha**2*x**2
    q=-3+alpha+2*alpha**2-alpha*x-3*alpha**2*x+4*alpha**2*x**2
    r=4+6*alpha+2*alpha**2-5*alpha*x-3*alpha**2*x+4*alpha**2*x**2
    d1=5-5*alpha+4*alpha*x
    d2=-7-5*alpha+4*alpha*x
    return [np.array([[p/d1,0],[q/d2,0]]),np.array([[0,-q/d1],[0,-r/d2]])]

def wing_order_zero(alpha, x):
    d1=5-5*alpha+4*alpha*x
    d2=7+5*alpha-4*alpha*x
    return [np.array([[d1/12]]),np.array([[d2/12]])]

# Strata: three hidden states and two tokens. alpha is the probability of remaining in the current state. While the
# state persists, state 0 emits token 0 with probability t0, state 1 with probability t1, and state 2 always emits
# token 1; every change of state emits token 1.
def strata_matrices(alpha, t0, t1):
    b = (1 - alpha) / 2
    return [
        np.array([[t0*alpha,0,0],[0,t1*alpha,0],[0,0,0]]),
        np.array([[(1-t0)*alpha,b,b],[b,(1-t1)*alpha,b],[b,b,alpha]]),
    ]

# The 1-HMM and the 0-HMM of Strata (Section 2.3).
def strata_order_one(alpha, t0, t1):
    n1=alpha*(t0**2+t1**2)
    n2=-t0+alpha*t0**2-t1+alpha*t1**2
    n3=3-2*alpha*t0+alpha**2*t0**2-2*alpha*t1+alpha**2*t1**2
    d1=t0+t1
    d2=-3+alpha*t0+alpha*t1
    return [np.array([[n1/d1,0],[alpha*n2/d2,0]]),np.array([[0,-n2/d1],[0,-n3/d2]])]

def strata_order_zero(alpha, t0, t1):
    p=alpha*(t0+t1)/3
    return [np.array([[p]]),np.array([[1-p]])]

