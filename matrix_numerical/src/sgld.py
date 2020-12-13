# GDA (Gradient descent-ascent.)
# x_{t+1} = numpy.clip(x_{t} + eta * f'(x_{t}), MIN_X, MAX_X,)


# OGDA (O Gradient descent-ascent.)
# x_{t+1} = numpy.clip(x_{t} + 2 * eta * f'(x_{t}) - eta * f'(x_{t-1}), MIN_X, MAX_X,)


# EG (Extra gradient)
# x_{t+0.5} = numpy.clip(x + eta * f'(x_{t}), MIN_X, MAX_X)
# x_{t+1} = numpy.clip(x + eta * f'(x_{t+0.5}), MIN_X, MAX_X)


# SGLD_DA (MixedNE-LD)
# In outer iteration t.
# Do x_{t+1} = SGLD_DA_STEP(x_{t}, eta, psi, beta, k)
# eta: learning rate, psi: some efficient added on the random noise (0.1),
# beta: momentum efficient (0.9), k: inner iteration (50).

# In SGLD_DA_STEP
# x_b = x_{t}
# x_ = x_{t}

# in each inner iteration.
# x__ = numpy.clip(x_ + eta * f'(x_) + np.sqrt(2 * eta) * np.random.normal(0, 1, 1) * psi, -2, 2)
# x_b = (1 - beta) * x_b + beta * x__
# x_ = x__

# x_{t+1} = (1 - beta) * x_{t} + beta * x_b



# def SGLD_step(x, y, eta, psi):
#     x_n = np.clip(x + eta * gx(x, y) + np.sqrt(2 * eta) * np.random.normal(0, 1, 1) * psi, -2, 2)
#     y_n = np.clip(y - eta * gy(x, y) + np.sqrt(2 * eta) * np.random.normal(0, 1, 1) * psi, -2, 2)
#     return x_n, y_n
#
#
# def SGLD_DA_step(x, y, eta, psi, beta, k):
#     x_ = x
#     y_ = y
#     x_b = x
#     y_b = y
#
#     for i in range(k):
#         x__, y__ = SGLD_step(x_, y_, eta, psi)
#         x_b = (1 - beta) * x_b + beta * x__
#         y_b = (1 - beta) * y_b + beta * y__
#         x_ = x__
#         y_ = y__
#
#     x_n = (1 - beta) * x + beta * x_b
#     y_n = (1 - beta) * y + beta * y_b
#
#     return x_n, y_n
#
#
# def SGLD_DA(x, y, eta=0.1, psi=0.1, beta=0.9, k=50, Max_iteration_time=1000):
#     dlist = np.zeros(Max_iteration_time)
#     dlistx = np.zeros(Max_iteration_time)
#     dlisty = np.zeros(Max_iteration_time)
#     d = dist(x, y)
#     dlist[0] = f(x, y)
#     dlistx[0] = x
#     dlisty[0] = y
#     for i in range(Max_iteration_time - 1):
#         x, y = SGLD_DA_step(x, y, eta, psi, beta, k)
#         d = dist(x, y)
#         dlistx[i + 1] = x
#         dlisty[i + 1] = y
#         d = f(x, y)
#         dlist[i + 1] = d
#     return dlist, dlistx, dlisty
