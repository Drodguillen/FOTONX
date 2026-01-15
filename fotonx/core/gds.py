from __future__ import annotations
import gdstk
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

LayerKey = Tuple[int, int]  # (layer, datatype)

@dataclass
class LayerInfo:
    layer: int
    datatype: int
    n_polys: int
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)

def read_gds_lib(path: str) -> gdstk.Library:
    return gdstk.read_gds(path)

def get_cell(lib: gdstk.Library, cell_name: str) -> gdstk.Cell:
    for c in lib.cells:
        if c.name == cell_name:
            return c
    raise KeyError(f"Cell '{cell_name}' not found. Available: {[c.name for c in lib.cells]}")

def read_gds(path: str, cell_name: Optional[str] = None) -> gdstk.Cell:
    lib = read_gds_lib(path)
    if cell_name is None:
        return lib.top_level()[0]          # compatibilidad (como lo tenías)
    return get_cell(lib, cell_name)        # <- aquí ignoras COMP

def list_layers(cell: gdstk.Cell) -> Dict[LayerKey, LayerInfo]:
    layers: Dict[LayerKey, LayerInfo] = {}

    for poly in cell.polygons:
        key: LayerKey = (int(poly.layer), int(poly.datatype))
        pts = poly.points
        xmin = float(pts[:, 0].min()); xmax = float(pts[:, 0].max())
        ymin = float(pts[:, 1].min()); ymax = float(pts[:, 1].max())

        if key not in layers:
            layers[key] = LayerInfo(key[0], key[1], 1, (xmin, ymin, xmax, ymax))
        else:
            info = layers[key]
            bx0, by0, bx1, by1 = info.bbox
            info.n_polys += 1
            info.bbox = (min(bx0, xmin), min(by0, ymin), max(bx1, xmax), max(by1, ymax))

    return layers

def print_layers(layers_dict):
    print("Layers found (layer, datatype): count, bbox")
    for key, info in sorted(layers_dict.items()):
        print(f"  {key}: n={info.n_polys}, bbox={info.bbox}")

def extract_polygons(cell: gdstk.Cell,
                     layer_map: Dict[str, Tuple[int, int]],
                     flatten: bool = True):
    work = cell
    if flatten:
        work = cell.copy(f"__tmp_flat_{cell.name}")
        work.flatten()

    out = {name: [] for name in layer_map}
    for poly in work.polygons:
        key = (int(poly.layer), int(poly.datatype))
        for name, (L, D) in layer_map.items():
            if key == (L, D):
                out[name].append(poly.points)
    return out




