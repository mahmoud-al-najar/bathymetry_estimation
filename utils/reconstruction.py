import numpy as np
from utils.preprocessing import apply_2d_gradient


def reconstruct_tile(model, full_tile, sub_tile_size=40, *preprocessing_callables):
    """
        model: trained sub_tile estimation model
        full_tile: full satellite image (cross_shore, long_shore, 4 bands)
        sub_tile_size: model input image dimensions. default: 40 (i.e. 40*40*4)
        preprocessing_callables: list of preprocessing functions to be called in order
    """
    cross_shore = full_tile.shape[0]
    long_shore = full_tile.shape[1]
    result = np.zeros((cross_shore - sub_tile_size, long_shore - sub_tile_size))

    c_margin = 0
    l_margin = 0
    if apply_2d_gradient in preprocessing_callables:
        c_margin += 1
        l_margin += 1

    count = 1
    for c in range(cross_shore - sub_tile_size - c_margin):
        print(count, ' - ', cross_shore - sub_tile_size)
        count += 1
        batch = np.empty((long_shore - sub_tile_size, sub_tile_size, sub_tile_size, 4))
        for l in range(long_shore - sub_tile_size - l_margin):
            sub_tile = full_tile[c:c + sub_tile_size + c_margin, l:l + sub_tile_size + l_margin, :]

            for f in preprocessing_callables:
                sub_tile = f(sub_tile)

            batch[l, ] = sub_tile

        r = model.predict(batch)
        result[c] = r.flatten() * 10

    return result
