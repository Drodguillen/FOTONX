import numpy as np

# All wavelengths in um. Sellmeier uses A in um^2.
MATERIAL_TABLE = {
    "sio2": {
        "A": (np.array([0.0684043, 0.1162414, 9.8961612], dtype=float) ** 2),
        "B": np.array([0.6961663, 0.4079426, 0.8974794], dtype=float),
        "n_nl": 2.7e-8,
    },
    "si": {
        "A": np.array([0.301516485**2, 1.13475115**2, 1104.0**2], dtype=float),
        "B": np.array([10.6684293, 0.0030434748, 1.54133408], dtype=float),
        "n_nl": 2.7e-8,
    },
    "si3n4": {
        "A": np.array([0.1353406**2, 1239.842**2], dtype=float),
        "B": np.array([3.0249, 40314.0], dtype=float),
        "n_nl": 2.7e-8,
    },
    "sf6": {
        "A": np.array([0.00857807248, 0.0420143003, 107.59306], dtype=float),
        "B": np.array([1.21640125, 0.13366454, 0.883399468], dtype=float),
        "n_nl": 0.0,
    },
    "air": {
        "A": np.array([], dtype=float),
        "B": np.array([], dtype=float),
        "n_nl": 0.0,
    },
}

