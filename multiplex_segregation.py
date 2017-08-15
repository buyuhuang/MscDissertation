import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import osmnx as ox
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize, linemerge
from shapely.wkt import loads
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import math
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import matplotlib.cm as cm

ox.config(log_file=True, log_console=True, use_cache=True)

#create helper functions
def boundary_from_areas(blocks, alpha = 1, buffer_dist = 0):
    """
    function to create boundary of multiple poligons
    """
    #get vertex of polygon
    g=blocks.geometry
    points = []
    #get points from polygons
    for poly in g:
        #check if geometry is polygon of Multipolygon
        if poly.type == 'Polygon':
            for pnt in poly.exterior.coords:
                points.append(geometry.Point(pnt))
        elif poly.type == 'MultiPolygon':
            for parts in poly:
                for pnt in parts.exterior.coords:
                    points.append(geometry.Point(pnt))

    def add_edge(edges, edge_points, coords, i,j):
        if (i,j) in edges or (j,i) in edges:
            return
        edges.add( (i,j) )
        edge_points.append(coords[ [i,j] ])
    
    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    
    #loop over triangles
    print('Calculating Boundary with alpha = ' + str(alpha))
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        #Lengths of sides of triangle 
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        #semiperimeter of triangle
        s = (a + b + c)/2.0
        #area and circumference
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        if area == 0:
            circum_r=0
        elif area >0:
            circum_r = a*b*c/(4.0*area)
        
        #radius filter
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    res = cascaded_union(triangles)
    if buffer_dist >0:
        res = res.buffer(buffer_dist)
    return gpd.GeoSeries(res)

def construct_street_graph(blocks, crs_osm, network_type, alpha, buffer_dist=0):
    """
    Function to create street network graph for a given area
    """
    print('Generating geometry')
    area = gpd.GeoDataFrame(geometry = boundary_from_areas(blocks, alpha = alpha, buffer_dist = buffer_dist))
    area.crs = crs_utm
    geometry = area.to_crs(crs_osm).unary_union
    
    print('Generating street graph')
    G = ox.graph_from_polygon(polygon = geometry, network_type = network_type, name = 'street_network')
#     G_undirected = nx.Graph(G)
    G.name = 'street'
    return G, area

##create function for plotting
def plot_network(G, area):
    fig, ax = ox.plot_graph(
        G,
        fig_height = 15,
        node_size = 10,
        node_zorder=1,
        edge_color = '#333333',
        edge_linewidth = 0.5,
        save = False, 
        show = False,
        close = False,
        bgcolor = 'w',
        filename = '{}'.format(G.name), 
        dpi=1200,
        equal_aspect = True
    )
    urban_outline = PolygonPatch(area.geometry[0], 
                                 fc='w', 
                                 ec='r', 
                                 linewidth=1, 
                                 alpha= 0.5,
                                 zorder= -1
                                )
    ax.add_patch(urban_outline)
    margin = 0.02
    west, south, east, north = area.geometry[0].bounds
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))
    plt.show()