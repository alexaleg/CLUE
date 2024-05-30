from clue import *
from sympy import QQ
from scipy import sparse

def to_sptf(M):
    import tensorflow as tf
    indices, values = M.to_coo()
    values = [float(v) for v in values]
    return tf.sparse.SparseTensor(indices, values, [M.nrows, M.ncols])

def to_scipy_sparse(M):
    rows, columns, values = M.to_coo()
    values = [float(v) for v in values]
    return sparse.coo_matrix((values, (rows, columns)), shape=(M.nrows, M.ncols))


M = SparseRowMatrix.from_list([[1,0],[0,1]], QQ)

print(M.to_coo())
print(to_scipy_sparse(M))

# J1 = tf.sparse.from_dense([[0, 0, 0,], [2, 0, -4 ], [-1,-1,0]],)
# J2 = tf.sparse.from_dense([[0, 2, 4.05,], [0, 0, 0 ], [0,0,0]],)
# J3 = tf.sparse.from_dense([[0, 4.05, 8,], [0, 0, 0 ], [0,0,0]],)
# L = tf.sparse.from_dense([[1.0, 0, 0]],)
# Lt = tf.sparse.transpose(L) 
# PL = tf.sparse.matmul(L, Lt)
# print(L, Lt,PL)



