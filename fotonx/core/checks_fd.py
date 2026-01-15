import numpy as np

def check_fd_operators(grid, ops, tol=0.15):
    Nx, Ny = grid.Nx, grid.Ny
    x, y = grid.x, grid.y
    X, Y = np.meshgrid(x, y, indexing="xy")

    Dx, Dy = ops["Dx"], ops["Dy"]
    Dxx, Dyy = ops["Dxx"], ops["Dyy"]

    interior = (slice(2, -2), slice(2, -2))

    # --- Test 1: f = x -> df/dx = 1
    f = X.copy()
    f_vec = f.ravel(order="C")
    dfdx_num = (Dx @ f_vec).reshape((Ny, Nx), order="C")
    err1 = np.mean(np.abs(dfdx_num[interior] - 1.0))

    # --- Test 2: f = y -> df/dy = 1
    f = Y.copy()
    f_vec = f.ravel(order="C")
    dfdy_num = (Dy @ f_vec).reshape((Ny, Nx), order="C")
    err2 = np.mean(np.abs(dfdy_num[interior] - 1.0))

    # --- Test 3: f = x^2 -> d2f/dx2 = 2
    f = X**2
    f_vec = f.ravel(order="C")
    d2fdx2_num = (Dxx @ f_vec).reshape((Ny, Nx), order="C")
    err3 = np.mean(np.abs(d2fdx2_num[interior] - 2.0))

    # --- Test 4: f = y^2 -> d2f/dy2 = 2
    f = Y**2
    f_vec = f.ravel(order="C")
    d2fdy2_num = (Dyy @ f_vec).reshape((Ny, Nx), order="C")
    err4 = np.mean(np.abs(d2fdy2_num[interior] - 2.0))

    ok = (err1 < tol) and (err2 < tol) and (err3 < tol) and (err4 < tol)

    return dict(ok=ok, err_dfdx=err1, err_dfdy=err2, err_d2fdx2=err3, err_d2fdy2=err4)
