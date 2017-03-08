
# coding: utf-8

# TSP solver for inspection predictions
# Developed by: BB
# TO DO:
    # Create D_mat class 
        # stores R_nodes 
        # provide weighted lengths
    # Create Tour class 
        # provide easier access to tour results
    # Better output for inspectors' use

import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import copy
import json
import sqlalchemy as sa
import matplotlib.pyplot as plt


class R_node():
    """Restaurant node class - stores node_id, r_id, risk weighting
    if no weight provided, defaults to 1
    """

    def __init__(self, r_id, n_id, weight=1):
        self.r_id = r_id
        self.n_id = n_id
        self.weight = weight


def find_closest(pos, latlng):
    """Get id of closest node to latlng
    pos : xy position of nodes
    latlng : xy position
    """
    close = np.argmin(np.sum((pos - latlng)**2, axis=1))
    return(close)


def mk_matrix(g, r_id, latlng):
    """Make distance matrix
    for restaurant latlng, find distance to all nodes
    return dictionary, plus node index
    """
    o = find_closest(pos, latlng)
    # Multiple restaurants per node
    if r_id not in g.node[o]['r_ids']:
        # Need to use this syntax, otherwise updates for all lists
        g.node[o]['r_ids'] = g.node.get(o)['r_ids'] + [r_id]
    d = nx.shortest_path_length(g, o, target=None, weight='length')
    return(d, o)


# Configure Boston network
bos = ox.graph_from_place('Boston, MA', network_type='drive')
# Relabel nodes sequential
bos_re = nx.convert_node_labels_to_integers(bos, first_label=0)
# Get positions of nodes
pos = np.array([(n[1]['x'], n[1]['y']) for n in bos.nodes(data=True)])
# Blank list appending r_ids
nx.set_node_attributes(bos_re, 'r_ids', [])


# Read in inspection info
df = pd.read_csv(insp_data_fp)
# Drop any with no lat/long
df = df[df.lat > 0]
# Unique id/name/lat/long
df = df[['restaurant_id', 'name', 'lat', 'lng']]
df.drop_duplicates(inplace=True)


engine = sa.create_engine('mssql+pymssql://UN:PW@SERVER/DB')
# Read in most recent risk score
preds = pd.read_sql_query("""
    select restaurant_id,
        max(predicted) as predicted,
        max(date) as date
    from rhipa_predictions_lists
    group by restaurant_id
    """, engine)


# Merge preds to unique lat/lng
df_preds = df.merge(preds, on='restaurant_id')


# Distance matrix as dict
D = {}
# Create full list of R_nodes
r_nodes = []
for r in df_preds.itertuples():
    D[r.restaurant_id], n_id = mk_matrix(bos_re,
                                         # latlng reversed in data
                                         r.restaurant_id, (r.lat, r.lng)[::-1])
    r_nodes.append(R_node(r.restaurant_id, n_id,
                          # Temporary - hard-coding weight (currently: 1/risk)
                          weight=1 / r.predicted)
                   )


# All nodes not connected to all nodes
# All but 9 restaurants connected to 99% of nodes, fine
all_nodes = .99 * len(bos_re.nodes())
cnt = 0
for d in D:
    if len(D[d]) < all_nodes:
        cnt += 1
print cnt, 'r_ids less than 99% connected'

# Any non-connected nodes, populate with 555k meters distance
for d in D:
    diff = set(bos_re.nodes()) - set(D[d].keys())
    for i in diff:
        D[d][i] = 555000


# Tour construction/tweaking
def randtour(rs, n, D):
    """Construct a random tour
    rs : R_node list
    n : size of tour
    """
    rand_r_nodes = list(np.random.choice(rs, size=n))
    return rand_r_nodes, length(rand_r_nodes, D)

# Adapted from TSPLIB
# (http://www.dcc.fc.up.pt/~jpp/code/gurobi_book/read_tsplib.py)


def nearest(last, unvisited, D):
    """Return the R_node which is closest to 'last'."""
    near = unvisited[0]
    min_dist = D[last.r_id][near.n_id] * near.weight
    for i in unvisited[1:]:
        try_dist = D[last.r_id][i.n_id] * i.weight
        if try_dist < min_dist:
            near = i
            min_dist = try_dist
    return near


def nearest_neighbor(unvisited, i, D):
    """Return tour starting from restaurant 'i', using the Nearest Neighbor.

    Nearest Neighbor heuristic:
    - start visiting restaurant i
    - while there are unvisited restaurants, follow to the closest one
    - return to restaurant i
    """

    unvisited.remove(i)
    last = i
    tour = [i]
    while unvisited != []:
        next = nearest(last, unvisited, D)
        tour.append(next)
        unvisited.remove(next)
        last = next
    return tour, length(tour, D)


def length(tour, D):
    """Calculate the length of a tour according to distance matrix 'D'."""

    z = 0
    for i in range(len(tour) - 1):
        # add distance * weight
        z += D[tour[i].r_id][tour[i + 1].n_id] * tour[i + 1].weight
    return z


def exchange(tour, tinv, i, j):
    """Exchange arcs (i,i+1) and (j,j+1) with (i,j) and (i+1,j+1).

    i, j : index of first,second node (will be swapped if i>j)

    for tour, remove the arcs (i,i+1) and (j,j+1) and
    insert (i,j) and (i+1,j+1).

    This is done by inverting the sublist of rs between i and j.
    """
    n = len(tour)
    # print i,j
    if i > j:
        i, j = j, i
    assert i >= 0 and j < n
    path = tour[i + 1:j + 1]
    path.reverse()
    tour[i + 1:j + 1] = path
    for k in range(i + 1, j + 1):
        tinv[tour[k]] = k


def mk_closest(D, tour):
    """Compute a sorted list of the distances for each of the nodes in the tour.
    This step is necessary for the local search

    For each node, the entry is in the form [(d1,n1), (d2,n2), ...]
    where each tuple is a pair (distance,node).
    """
    C = {}
    for r in tour:
        dlist = [
            (D[r.r_id][r2.n_id] * r2.weight, r2)
            for r2 in tour if r2 != r
        ]
        dlist.sort()
        C[r] = dlist
    return C


def improve(tour, z, D, C):
    """Try to improve tour 't' by exchanging arcs; return improved tour length.

    If possible, make a series of local improvements on the solution 'tour',
    using a breadth first strategy, until reaching a local optimum.
    """
    n = len(tour)
    tinv = {}
    for k in range(n):
        tinv[tour[k]] = k  # position of each city in 't'
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        dist_ab = D[a.r_id][b.n_id] * b.weight
        improved = False
        # Run through sorted list of distances
        for dist_ac, c in C[a]:
            # If reach point where ac longer than ab, break
            if dist_ac >= dist_ab:
                break
            # Otherwise, find where c is in the tour
            j = tinv[c]
            # If c is already connected to a (cab), move on
            if i - j == 1:
                continue
            d = tour[(j + 1) % n]
            # Test a change
            dist_cd = D[c.r_id][d.n_id] * d.weight
            dist_bd = D[b.r_id][d.n_id] * d.weight
            delta = (dist_ac + dist_bd) - (dist_ab + dist_cd)
            if delta < 0:       # exchange decreases length
                exchange(tour, tinv, i, j)
                z += delta
                improved = True
                break
        if improved:
            continue
        for dist_bd, d in C[b]:
            if dist_bd >= dist_ab:
                break
            # If d comes before b (ab = db), move on
            if i - tinv[d] == 1:
                continue
            j = tinv[d] - 1
            if j == -1:
                j = n - 1
            c = tour[j]
            dist_cd = D[c.r_id][d.n_id] * d.weight
            dist_ac = D[a.r_id][c.n_id] * c.weight
            delta = (dist_ac + dist_bd) - (dist_ab + dist_cd)
            if delta < 0:       # exchange decreases length
                exchange(tour, tinv, i, j)
                z += delta
                break
    return tour


def localsearch(tour, z, D, C=None):
    """Obtain a local optimum starting from solution t; return solution length.

    Parameters:
      tour : initial tour
      z : length of the initial tour
      D : distance matrix
      C : (optional) sorted list of distance, created otherwise
    """
    if C is None:
        # create a sorted list of distances to each node
        C = mk_closest(D, tour)
    while 1:
        tour = improve(tour, z, D, C)
        newz = length(tour, D)
        if newz < z:
            z = newz
        else:
            break
    return tour, z


# Utilites for output
def add_tour(name, tour):
    """ Create tour results for finding max and output
    name : name of the tour
    """
    print '{} tour: {}'.format(name, tour[1])
    return({name:
            {'order': tour[0],
             'length': tour[1]}
            }
           )


def find_path(g, tour):
    """
    Finds full tour path
    """
    path = []
    for i in range(len(tour) - 1):
        try:
            path += nx.shortest_path(g, tour[i].n_id,
                                     tour[i + 1].n_id, weight='length')
        except nx.NetworkXNoPath:
            pass
        # Remove last place to avoid duplicate
        last = [path.pop(-1)]
    # Add on last if end of loop
    path += last
    return(path)


def graph_tour(tour, g):
    """ Maps tour, labelled for sequence """
    path = find_path(g, tour)
    tour_nodes = [n.n_id for n in tour]
    order_dict = dict((x, i) for i, x in enumerate(tour_nodes))
    pos = zip(nx.get_node_attributes(g, 'x').values(),
              nx.get_node_attributes(g, 'y').values())

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(g, pos=pos, node_size=1, node_color='gray', alpha=.1, ax=ax)
    nx.draw_networkx_nodes(g, pos=pos, nodelist=tour_nodes, node_size=100,
                           node_color='b', ax=ax)
    nx.draw_networkx_labels(g, pos=pos, labels=order_dict, font_color='w')
    nx.draw_networkx_edges(g, pos=pos, edgelist=zip(path[:-1], path[1:]),
                           edge_color='r', ax=ax)
    return(fig, ax)


def output(tour_results, g):
    """Outputs the shortest trip order (r_ids) and map"""
    # Get shortest trip
    shortest = tour_results[0]
    for r in tour_results[1:]:
        if r[1] < shortest[1]:
            shortest = r
    # Get r_id order
    order_ids = [int(r.r_id) for r in shortest[2]]
    # Output to json
    name = shortest[0]
    with open(name + '.json', 'w') as f:
        json.dump({name:
                   {'length': shortest[1],
                    'order': order_ids}
                   }, f)

    # Make figure, save
    fig, ax = graph_tour(shortest[2], g)
    fig.savefig(name + '.png', bbox_inches='tight')


if __name__ == "__main__":
    tour_results = []

    rand_tour, z = randtour(r_nodes, 10, D)
    tour_results.append(['random', z, rand_tour])
    print tour_results[-1][0], tour_results[-1][1]

    near_tour, z = nearest_neighbor(copy.copy(rand_tour), rand_tour[0], D)
    tour_results.append(['nearest', z, near_tour])
    print tour_results[-1][0], tour_results[-1][1]

    local_tour, z = localsearch(copy.copy(near_tour), z, D)
    tour_results.append(['local', z, local_tour])
    print tour_results[-1][0], tour_results[-1][1]

    output(tour_results, bos_re)
