import numpy as np
from scipy import sparse
import pandas as pd


count = 100000
high = 500

indices = np.empty(shape=(count,), dtype=[("row", np.int32), ("col", np.int32)])
indices["row"] = np.random.randint(low=0, high=high, size=count)
indices["col"] = np.random.randint(low=0, high=high, size=count)

values = np.random.random(size=count)


def aggregate_with_sparse(values, rows, cols, count):
    matrix = sparse.coo_matrix((values, (rows, cols)), (count, count)).tocsr().tocoo()
    return matrix


matrix = aggregate_with_sparse(values, indices["row"], indices["col"], count)

matrix.row.shape, matrix.col.shape, matrix.data.shape


# %timeit aggregate_with_sparse(values, indices['row'], indices['col'], count)

# Takes about 8 ms


def with_pandas(indices, values):
    df = pd.concat(
        [pd.Series(indices["row"]), pd.Series(indices["col"]), pd.Series(values)],
        axis=1,
    )
    df.columns = ["row", "col", "values"]
    result = df.groupby(["row", "col"]).sum()
    return (
        ind.get_level_values("row").to_numpy(),
        ind.get_level_values("col").to_numpy(),
        result.to_numpy(),
    )


# %timeit with_pandas(indices, values)

# Takes about 27.7 ms
