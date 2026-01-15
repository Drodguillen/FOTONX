from fotonx.core.gds import read_gds_lib, get_cell, extract_polygons
from fotonx.core.raster import make_grid_from_polys, rasterize_mat_id

def build_from_gds(geom_cfg, grid_cfg):
    lib = read_gds_lib(geom_cfg.gds_path)
    cell = get_cell(lib, geom_cfg.cell_name)
    polys = extract_polygons(cell, geom_cfg.layer_map, False)

    grid = make_grid_from_polys(polys, dx=grid_cfg.dx, dy=grid_cfg.dy,
                                pad_x=grid_cfg.pad_x, pad_y=grid_cfg.pad_y)
    mat_id = rasterize_mat_id(grid, polys, geom_cfg.mat_ids, background_name=geom_cfg.background_name)
    return polys, grid, mat_id
