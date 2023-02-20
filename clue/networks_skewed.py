import requests, zipfile, io, logging, pandas

from sympy import QQ, CoercionFailed

from .clue import FODESystem
from .linalg import SparseRowMatrix
from .rational_function import SparsePolynomial

logger = logging.getLogger(__name__)

def FromNetwork(network, name=None, column = -1, undirected=None, adjacency=True, coercion_error=False, **kwds):
    logger.info(f"[FromNetwork] Reaching the website for the model {network}_{name}...",)
    r = requests.get(f"https://networks.skewed.de/api/net/{network}").json()
    name = r["nets"][0] if name is None else name
    if not name in r["nets"]:
        raise ValueError(f"{name} is not valid for the class {network}")
        
    ## Getting other inputs (undirected and multigraph)
    undirected = undirected if undirected != None else (not (r["analyses"]["is_directed"] if len(r['nets']) == 1 else r["analyses"][name]["is_directed"]))
        
    logger.info(f"[FromNetwork] Reaching the website for the csv description...")          
    graph_zip = zipfile.ZipFile(io.BytesIO(requests.get(f"https://networks.skewed.de/net/{network}/files/{name}.csv.zip").content))
    
    logger.info(f"[FromNetwork] Reading vertices...")     
    vertices = pandas.read_csv(io.BytesIO(graph_zip.read("nodes.csv")), delimiter=",")
    logger.info("[FromNetwork] Reading edges...")
    edges = pandas.read_csv(io.BytesIO(graph_zip.read("edges.csv")), delimiter=",")
    logger.info("[FromNetwork] Creating variable names...")
    name_index = None
    for i,column_name in enumerate(vertices.columns):
        if any(pos_name in column_name for pos_name in ("name", "label", "meta")):
            name_index=i; break
    
    varnames = vertices[vertices.columns[name_index]].tolist() if name_index != None else [f"S{i}" for i in range(len(vertices))]
    if len(set(varnames)) < len(varnames): # repeated names --> better use generic
        logger.warning(f"[FromNetwork] Found repeated names in the vertices. Using generic names")
        varnames = [f"S{i}" for i in range(len(vertices))]

    logger.info(f"[FromNetwork] Building {'adjacency' if adjacency else 'laplacian'} matrix...")
    equations = SparseRowMatrix(len(varnames))
    if len(edges.columns) < 2: raise TypeError("The edges csv file is in incorrect format: too few columns")
    elif len(edges.columns) == 2: # only two columns: always adding 1
        for (src, trg) in edges.itertuples(index=False, name=None):
            equations.increment(src, trg, 1)
    else: # more columns: adding the given column
        for (src, trg, val) in edges[[edges.columns[0], edges.columns[1], edges.columns[column]]].itertuples(index=False, name=None):
            if val != 0: 
                try: 
                    equations.increment(src, trg, QQ.convert(val))
                except CoercionFailed as error:
                    if not coercion_error:
                        logger.warning(f"[FromNetwork] Error in casting to rational: {val}")
                        equations.increment(src, trg, 1)
                    else:
                        raise error
    if not adjacency: # we need the laplacian matrix
        for i in equations.nonzero:
            row = equations[i]; degree = 0
            for j in row.nonzero:
                degree += row[j]
            equations.increment(i,i, -degree)
    logger.info(f"[FromNetwork] Building differential system...")
    system = FODESystem.LinearSystem(equations, variables=varnames,name=f"{network}_{name}",**kwds)
    logger.info("[FromNetwork] Returning differential system")
    return system