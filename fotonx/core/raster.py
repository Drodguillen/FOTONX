import numpy as np
from fotonx.core.grid import Grid2D

def bbox_from_polys(polys: dict[str, list[np.ndarray]]):
    xmin = +np.inf
    ymin = +np.inf
    xmax = -np.inf
    ymax = -np.inf
    for _, plist in polys.items():
        for pts in plist:
            xmin = min(xmin, float(pts[:,0].min()))
            xmax = max(xmax, float(pts[:,0].max()))
            ymin = min(ymin, float(pts[:,1].min()))
            ymax = max(ymax, float(pts[:,1].max()))
    return xmin, ymin, xmax, ymax

def make_grid_from_polys(polys, dx, dy, pad_x=0.0, pad_y=0.0):
    xmin, ymin, xmax, ymax = bbox_from_polys(polys)
    xmin -= pad_x; xmax += pad_x
    ymin -= pad_y; ymax += pad_y

    Sx = xmax - xmin
    Sy = ymax - ymin

    Nx = int(np.round(Sx/dx)) + 1
    Ny = int(np.round(Sy/dy)) + 1

    x = xmin + dx*np.arange(Nx)
    y = ymin + dy*np.arange(Ny)

    return Grid2D(x=x, y=y, dx=dx, dy=dy)

def make_grid_from_window(xmin, xmax, ymin, ymax, dx, dy):
    Sx = xmax - xmin
    Sy = ymax - ymin

    Nx = int(np.round(Sx/dx)) + 1
    Ny = int(np.round(Sy/dy)) + 1

    x = xmin + dx*np.arange(Nx)
    y = ymin + dy*np.arange(Ny)

    return Grid2D(x=x, y=y, dx=dx, dy=dy)

def point_in_polygon(x, y, poly):
    inside = False
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        if (y0 > y) != (y1 > y):
            x_int = x0 + (y - y0)*(x1 - x0) / (y1 - y0)
            if x_int > x:
                inside = not inside
        x0, y0 = x1, y1
    return inside

def rasterize_mat_id(grid: Grid2D, polys: dict[str, list[np.ndarray]], name_to_id: dict[str,int], background_name):
    Ny, Nx = grid.Ny, grid.Nx
    mat_id = np.zeros((Ny, Nx), dtype=np.uint16)
    bg = int(name_to_id[background_name])
    mat_id = np.full((Ny, Nx), bg, dtype=np.uint16)

    for name, plist in polys.items():
        mid = int(name_to_id[name])
        for poly in plist:
            # cheap bbox to reduce point tests
            xmin = poly[:,0].min(); xmax = poly[:,0].max()
            ymin = poly[:,1].min(); ymax = poly[:,1].max()

            # indices in grid that overlap bbox
            ix = np.where((grid.x >= xmin) & (grid.x <= xmax))[0]
            iy = np.where((grid.y >= ymin) & (grid.y <= ymax))[0]
            if ix.size == 0 or iy.size == 0:
                continue

            for j in iy:
                yj = float(grid.y[j])
                for i in ix:
                    xi = float(grid.x[i])
                    if point_in_polygon(xi, yj, poly):
                        mat_id[j,i] = mid

    return mat_id

