################################################################################
# Module: multiplex_segregation.py
# Description: Retrieves and constructs multiplex network from bus, tram and street network
# 			   for the city of Cuenca, Ecuador. Multiplex is used to calculate segregation
#              using ICV index from socio economic data using random walks.
#              Scripts developed as part of MSc. in Smart
#              Cities and Urban Analytics Dissertation 
# License: MIT, see full license in LICENSE.txt
# Web: mateoneira.github.io
################################################################################


import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import osmnx as ox
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize, linemerge
from shapely.wkt import loads
import scipy as sp
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import math
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import matplotlib.cm as cm
from geopandas.tools import overlay
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
import matplotlib.colors as colors
import seaborn as sns
from scipy.sparse import identity, spdiags, linalg
import time

ox.config(log_file=True, log_console=True, use_cache=True)

def boundary_from_areas(blocks, alpha = 1, buffer_dist = 0):
    """
    Creates spatial boundary given unconnected block area geometries of a city
    using alpha shape

    Parameters
    ----------
    	blocks: geodataframe containing city block shape geometry
    	alpha: alpha value to calculate alpha shape of boundary area
    	buffer_dist: buffer distance to create boundary (in meters)

	Returns
    ----------
    	boundary: geopandas geoseries 
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
    print('Calculating Boundary with alpha = {} ...'.format(alpha))
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

def construct_street_graph(blocks, crs_osm, crs_utm, alpha, buffer_dist=0, speed=5):
    """
    Creates street network graph from OSM data within the spatial boundaries
    of the blocks polygon

    The street network is created as a time weighted multidigraph, 
    all streets modelled as bi-directional edges to represent a pedestrian network.

    Parameters
    ----------
    	blocks: geodataframe containing city block shape geometry
    	crs_osm: projection system of Open Street Map
    	crs_utm: projection system of blocks
    	alpha: alpha value to calcualte alpha shape of boundary area
    	buffer_dist: buffer distance to retrieve network (in meters)
    	speed: speed for calculating weights of edges (km/h)

	Returns
    ----------
    	street_network: networkx multidigraph
    	area: geopandas geoseries 
    """
    network_type = 'drive'  
    print('Generating geometry...')
    area = gpd.GeoDataFrame(geometry = boundary_from_areas(blocks, alpha = alpha, buffer_dist = buffer_dist))
    area.crs = crs_utm
    geoms = area.to_crs(crs_osm).unary_union
    
    print('Generating street graph...')
    G = ox.graph_from_polygon(polygon = geoms, network_type = network_type, name = 'street_network')
    print('Reprojecting street graph...')
    G = ox.project_graph(G, to_crs=crs_utm)
    G = to_undirected(G)
    G = to_time_weighted(G, speed)

    node_id = [u for u in G.nodes(data=False)]
    geoms = [geometry.Point(data['x'], data['y']) for u, data in G.nodes(data=True)]
    node_dict= dict(zip(node_id, geoms))

    nx.set_node_attributes(G, 'geometry', node_dict)

    G.name = 'street'
    print('done!')
    return G, area

def to_undirected(G):
    #make undirected, while conserving multidigraph structure
    edges = []
    for u, v, keys, data in G.edges(data=True, keys=True):
        if data['oneway']:
            data['oneway']= False
            G[u][v][keys]['oneway'] = False
            edges.append((v,u,data))
    G.add_edges_from(edges)
    return G

def to_time_weighted(G, speed):
    for u, v, keys, data in G.edges(data='length', keys=True):
        UJT = 1/(speed * 16.666666666667) #turn km/h to min/meter
        G[u][v][keys]['weight'] = data * UJT
    return G


def join_lines(line, line_list):
    """
    Joins multiline geometries and returns a linstring object through recursion

    Parameters
    ----------
        line: list containing list of point geometries of coordinates of linestrings contained within multilinestring 
        line_list: list containing coordinates of linestring contained within multilinestring

    Returns
    ----------
        single array of coordinates defining joined line: array
    """
    p1 = geometry.Point(line[-1])
    list_ = []
    lList = []
    if line_list is not None:
        for l in line_list:
            list_.append(l)
            lList.append(l)
            list_.append(list(reversed(l)))
        for l in list_:
            #check if end_point of line is equal to point of other
            p2 = geometry.Point(l[0])
            if p1.distance(p2)<21:
                line_list = lList.remove(l)
                for coord in l:
                    line.append(coord)
                join_lines(line, lList)
            else: 
                return line

def clean_stops(stops, tolerance = 50):
    """
    Joins bus stops that are within a tolerance distance and returns centroid

    Parameters
    ----------
        stops: geopandas object containing stop geometries
        tolerance: distance tolerance in meters

    Returns
    ----------
        stops: geoseries
    """
    buffered_stops = stops.buffer(tolerance).unary_union
    if isinstance(buffered_stops, geometry.Polygon):
        buffered_stops = [buffered_stops]
    unified_stops = gpd.GeoSeries(list(buffered_stops))
    stops_centroids = unified_stops.centroid
    return stops_centroids

def clean_lines(busLineGPD):
    """
    Creates geodataframe containing geometries of LineString objects

    Parameters
    ----------
        busLineGPD: geopandasdataframe containing buslines geometries

    Returns
    ----------
        lines: geodataframe
    """
    lines_list = []
    for lineNum in busLineGPD.LINEA.unique():
        bus_lines = busLineGPD[busLineGPD.LINEA == lineNum]
        for i, bus_line in bus_lines.iterrows():
            line = bus_line.geometry
            #check if line is multilinestring to store as multiple singlelinesstrings
            if isinstance(line, geometry.MultiLineString):
                lgeos = line.geoms
                lines = []
                for tl in lgeos:
                    tlines = []
                    if tl.length > 20 and tl.coords[0] != tl.coords[-1]:
                        coord = list(tl.coords)
                        for j in range(len(coord)):
                            point = coord[j]
                            tlines.append(point)
                        lines.append(coord)
                #choose first line and look for continuation
                lineCoord = lines[0]
                lineList = lines[1:]
                lineJoin = join_lines(lineCoord, lineList)
                lineJoin = join_lines(list(reversed(lineJoin)), lineList)
                tlines = geometry.LineString(coor for coor in lineJoin)
                linesGPD = gpd.GeoDataFrame({'nline': [bus_line.LINEA],
                                             'way': [bus_line.TRAYECTO],
                                             'route': [bus_line.RUTA],
                                             'geometry': [tlines],
                                             'ngeom': [1],
                                           })
            else:
                linesGPD = gpd.GeoDataFrame({'nline': [bus_line.LINEA],
                                             'way': [bus_line.TRAYECTO],
                                             'route': [bus_line.RUTA],
                                             'geometry': [line],
                                             'ngeom': [1]
                                           })

            lines_list.append(linesGPD)
    res = gpd.GeoDataFrame(pd.concat(lines_list, ignore_index=True))
    return res
    

def snap_stops_to_lines(busLineGPD, busStopsGPD, area, tolerance = 50):
    """
    Snaps points to lines based on tolerance distance and line number
    Returns a GPD with stops along with their line number and way

    Parameters
    ----------
        busLineGPD: geopandasdataframe containing bus lines geometries
        busStopsGPD: geopandasdataframe containing bus stop geometries
        area: geoseries of boundary polygon
        tolerance: distance tolerance for snapping points (in meters)

    Returns
    ----------
        stops: geodataframe
    """
    stop_list = []
    for lineNum in busLineGPD.nline.unique():
        bus_lines = busLineGPD[busLineGPD.nline == lineNum]
        bus_stops = busStopsGPD[busStopsGPD.lineNum == lineNum]

        #snap points to line
        for i, bus_line in bus_lines.iterrows():
            line = bus_line.geometry

            #get only points within buffer and inside area
            line_buff = line.buffer(tolerance)
            stops = bus_stops[bus_stops.intersects(line_buff)]
            stops = stops[stops.intersects(area.geometry[0])]
            stops = clean_stops(stops, tolerance)
            points_snap = [line.project(stop) for stop in stops.geometry]
            snapped_points = [line.interpolate(point) for point in points_snap]
            pointsGPD = gpd.GeoDataFrame({'nline': [lineNum for i in range(len(snapped_points))],
                                    'way': [bus_line.way for i in range(len(snapped_points))],
                                    'route': [bus_line.route for i in range(len(snapped_points))],
                                    'geometry': snapped_points,
                                    'ngeom': [bus_line.ngeom for i in range(len(snapped_points))],
                                    'lgth': points_snap,
                                    'x': [point.xy[0][0] for point in snapped_points],
                                    'y': [point.xy[1][0] for point in snapped_points]
                                   })
            stop_list.append(pointsGPD)
    res = gpd.GeoDataFrame(pd.concat(stop_list, ignore_index=True))
    #remove duplicates and na
    res = res.drop_duplicates(subset=['nline', 'way', 'route', 'lgth'])
    res = res.dropna(how = 'all')
    #give unique id to points
    res['id']= [id_ for id_ in range(len(res))]
    return res

def create_transfers(G, weight = 15):
    """
    Creates transfer edges based on proximity 

    Parameters
    ----------
        G: multidigraph of transport network
        weight: time between transfers in minutes

    Returns
    ----------
        G: multidigraph 
    """
    stops_geoms = pd.DataFrame.from_dict(G.node, orient='index')
    stops_geoms = gpd.GeoDataFrame(stops_geoms)
    stops_buff = stops_geoms.buffer(15).unary_union
    unified_stops = gpd.GeoSeries(list(stops_buff))

    #create edge for all points within buffer
    for buff in unified_stops:
        stops = stops_geoms[stops_geoms.intersects(buff)]
        #replace x, y coord  and add edge
        for i in stops.id:
            G.node[i]['x'] = buff.centroid.xy[0][0]
            G.node[i]['y'] = buff.centroid.xy[1][0]
            G.node[i]['geometry'] = buff.centroid
            for j in stops.id:
                if i != j:
                    G.add_edge(u = i,
                               v = j, 
                               way = 'transfer',
                               key = 0,
                               weight = weight,
                               osmid = np.nan
                              )
    return G

def snap_lines_to_points(G):
    """
    Snaps begining and end of lines to nodes geometry

    Parameters
    ----------
        G: multidigraph of transport network
        weight: time between transfers in minutes

    Returns
    ----------
        G: multidigraph 
    """
    for u, v, keys, line in G.edges(data='geometry', keys=True):
        if isinstance(line, geometry.LineString):
            lcoords = list(line.coords)
            start = [(G.node[u]['x'], G.node[u]['y'])]
            end = [(G.node[v]['x'], G.node[v]['y'])]
            sPoint = geometry.Point(lcoords[0])
            sPoint_ = geometry.Point((G.node[u]['x'], G.node[u]['y']))
            if sPoint.distance(sPoint_) <50:
                if len(lcoords)>2:
                    coords = start + lcoords[1:len(lcoords)-1] + end
                else:
                    coords = start + lcoords[1:len(lcoords)-1] + end    
            else:
                if len(lcoords)>2:
                    coords = end + lcoords[1:len(lcoords)-1] + start
                else:
                    coords = end + lcoords[1:len(lcoords)-1] + start
            line = geometry.LineString(coords)
            G[u][v][keys]['geometry'] = line
    return G

    
def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [geometry.LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(geometry.Point(p))
        if pd == distance:
            return [
                geometry.LineString(coords[:i+1]),
                geometry.LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                geometry.LineString(coords[:i] + [(cp.x, cp.y)]),
                geometry.LineString([(cp.x, cp.y)] + coords[i:])]
        
def create_bus_network(linesGPD, stopsGPD, area, crs_utm = {'init':'epsg:32717'}, speed = 30):
    """
    Creates bus network graph from line and stop shapefile data 
    within the spatial boundaries of the blocks polygon

    The bus network is created as a time weighted multidigraph, 
    all bus lines are modeled as bi-directional edges to represent the network.

    transfers between stations area modeled as time weighted edges.

    Parameters
    ----------
	linesGPD: geodataframe containing line geometries
	stopsGPD: geodataframe containing stop geometries
	speed: speed for calculating weights of edges (km/h)

	Returns
    ----------
	networkx multidigraph
    """
    linesGPD = clean_lines(linesGPD)
    stopsGPD = snap_stops_to_lines(linesGPD, stopsGPD, area)
    UJT = 1/(speed * 16.666666666667) #turn km/h to min/meter
    
    line_list = []
    for lineNum in linesGPD.nline.unique():
            bus_lines = linesGPD[linesGPD.nline == lineNum]
            bus_stops = stopsGPD[stopsGPD.nline == lineNum]


            #cut lines at points
            for i, bus_line in bus_lines.iterrows():
                line = bus_line.geometry
                stop = bus_stops[bus_stops.route == bus_line.route]
                stop_ = stop[stop.ngeom == bus_line.ngeom]
                stop_sorted = stop_.sort_values(by = "lgth").reset_index()
                stops_c = stop.sort_values(by = "lgth").reset_index()
                if len(stop_sorted) is not 0:
                    for point in range(len(stop_sorted)-1):
                        pId = stop_sorted.id[point]
                        pId2 = stop_sorted.id[point+1]
                        dist1 = stop_sorted.lgth[point]
                        dist2 = stop_sorted.lgth[point+1]
                        lgth = dist2 - dist1
                        cut1 = cut(line, dist1)
                        if len(cut1) > 1:
                            tLine = cut1[1]
                        else:
                            tLine = cut1[0]
                        cut2 = cut(tLine, lgth)[0]
                        #only save if line starts and ends on stops
                        if (stop_sorted.geometry[point].distance(geometry.Point(cut2.coords[0]))<1) and (stop_sorted.geometry[point+1].distance(geometry.Point(cut2.coords[-1]))<1):
                            edgeGPD = gpd.GeoDataFrame({'nline': [lineNum, lineNum],
                                                        'way': [bus_line.way, bus_line.way],
                                                        'route': [bus_line.route, bus_line.route],
                                                        'geometry': [cut2, cut2],
                                                        'ngeom': [bus_line.ngeom, bus_line.ngeom],
                                                        'length': [lgth, lgth],
                                                        'u': [pId, pId2],
                                                        'v': [pId2, pId],
                                                        'from': [cut2.coords[0], cut2.coords[-1]],
                                                        'to': [cut2.coords[-1], cut2.coords[0]],
                                                        'weight': [lgth*UJT, lgth*UJT],
                                                        
                                                       })
                            line_list.append(edgeGPD)
    edge_list = gpd.GeoDataFrame(pd.concat(line_list, ignore_index=True))
    edge_list['key'] = [i for i in range(len(edge_list))]
    edge_list['osmid'] = [i for i in range(len(edge_list))]
    node_list = stopsGPD
    node_list.crs = crs_utm
    edge_list.crs = crs_utm
    node_list['osmid'] = [i for i in range(len(node_list))]
    node_list.gdf_name = 'Nodes_list'
    
    # plot to check each line plotted correctly
    print('plotting results for each line:')
    for lineNum in linesGPD.nline.unique():
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,3))
        fig.suptitle('Line: {}'.format(lineNum), fontsize=12, fontweight='bold')
        ax.set_aspect("equal")
        stopsGPD[stopsGPD.nline==lineNum].plot(ax=ax, color = 'red')
        edge_list[edge_list.nline==lineNum].plot(ax=ax, color = 'grey')
        plt.show()
    
    #create network
    G = ox.gdfs_to_graph(node_list, edge_list)
    G = create_transfers(G)
    G = snap_lines_to_points(G)
    G.name = 'bus'
    return(G)

def clean_lines_tram(tramLineGPD):
    """
    Make geodataframe of lines that contains only single line strings
    and make lineString Z into linestring
    """
    lines_list = []
    for i, tram_line in tramLineGPD.iterrows():
        line = tram_line.geometry
        #check if line is multilinestring to store as multiple singlelinesstrings
        if isinstance(line, geometry.MultiLineString):
            lgeos = line.geoms
            lines = []
            for tl in lgeos:
                tlines = []
                if tl.length > 20 and tl.coords[0] != tl.coords[-1]:
                    coord = list(tl.coords)
                    for j in range(len(coord)):
                        point = coord[j][0:2]
                        tlines.append(point)
                    lines.append(coord)
            #choose first line and look for continuation
            lineCoord = lines[0]
            lineList = lines[1:]
            lineJoin = join_lines(lineCoord, lineList)
            lineJoin = join_lines(list(reversed(lineJoin)), lineList)
            tlines = geometry.LineString(coor for coor in lineJoin)
            linesGPD = gpd.GeoDataFrame({'way': [tram_line.Layer],
                                         'geometry': [tlines],
                                         'ngeom': [1],
                                       })
        else:
            line_list = []
            coord = list(line.coords)
            for j in range(len(coord)):
                point = coord[j][0:2]
                line_list.append(point)
            line = geometry.LineString(coor for coor in line_list)                
            linesGPD = gpd.GeoDataFrame({'way': [tram_line.Layer],
                                         'geometry': [line],
                                         'ngeom': [1]
                                       })

        lines_list.append(linesGPD)
    res = gpd.GeoDataFrame(pd.concat(lines_list, ignore_index=True))
    return res    

def snap_stops_to_lines_tram(tramLine, tramStops, area, tolerance = 50):
    """
    Snaps points to lines based on tolerance distance
    """
    stop_list = []

        #snap points to line
    for i, tram_line in tramLine.iterrows():
        line = tram_line.geometry

        #get only points within buffer and inside area
        line_buff = line.buffer(tolerance)
        stops = tramStops[tramStops.intersects(line_buff)]
        stops = stops[stops.intersects(area.geometry[0])]
        stops = clean_stops(stops, tolerance)
        points_snap = [line.project(stop) for stop in stops.geometry]
        snapped_points = [line.interpolate(point) for point in points_snap]
        pointsGPD = gpd.GeoDataFrame({'way': [tram_line.way for i in range(len(snapped_points))],
                                    'geometry': snapped_points,
                                    'ngeom': [tram_line.ngeom for i in range(len(snapped_points))],
                                    'lgth': points_snap,
                                    'x': [point.xy[0][0] for point in snapped_points],
                                    'y': [point.xy[1][0] for point in snapped_points]
                                       })
        stop_list.append(pointsGPD)
    res = gpd.GeoDataFrame(pd.concat(stop_list, ignore_index=True))
    #remove duplicates and na
    res = res.drop_duplicates(subset=['way', 'lgth', 'x', 'y'])
    res = res.dropna(how = 'all')
    #give unique id to points
    res['id']= [id_ for id_ in range(len(res))]
    return res
        
def create_tram_network(linesGPD, stopsGPD, area, crs_utm = {'init':'epsg:32717'},speed = 40):
    """
    Create tram network from tram lines and tram stop geometries
    """
    linesGPD = clean_lines_tram(linesGPD)
    stopsGPD = snap_stops_to_lines_tram(linesGPD, stopsGPD, area)
    UJT = 1/(speed * 16.666666666667) #turn km/h to min/meter
    
    line_list = []

    #cut lines at points
    for i, tram_line in linesGPD.iterrows():
        line = tram_line.geometry
        stop_ = stopsGPD[stopsGPD.way == tram_line.way]
        stop_sorted = stop_.sort_values(by = "lgth").reset_index()
        if len(stop_sorted) is not 0:
            for point in range(len(stop_sorted)-1):
                pId = stop_sorted.id[point]
                pId2 = stop_sorted.id[point+1]
                dist1 = stop_sorted.lgth[point]
                dist2 = stop_sorted.lgth[point+1]
                lgth = dist2 - dist1
                cut1 = cut(line, dist1)
                if len(cut1) > 1:
                    tLine = cut1[1]
                else:
                    tLine = cut1[0]
                cut2 = cut(tLine, lgth)[0]
                #only save if line starts and ends on stops
                if (stop_sorted.geometry[point].distance(geometry.Point(cut2.coords[0]))<1) and (stop_sorted.geometry[point+1].distance(geometry.Point(cut2.coords[-1]))<1):
                    edgeGPD = gpd.GeoDataFrame({'way': [tram_line.way],
                                                'geometry': [cut2],
                                                'ngeom': [tram_line.ngeom],
                                                'length': [lgth],
                                                'u': [pId],
                                                'v': [pId2],
                                                'from': [cut2.coords[0]],
                                                'to': [cut2.coords[-1]],
                                                'weight': [lgth*UJT]
                                               })
                    line_list.append(edgeGPD)
    edge_list = gpd.GeoDataFrame(pd.concat(line_list, ignore_index=True))
    edge_list['key'] = [i for i in range(len(edge_list))]
    edge_list['osmid'] = [i for i in range(len(edge_list))]
    node_list = stopsGPD
    node_list['osmid'] = [i for i in range(len(node_list))]
    node_list.gdf_name = 'Nodes_list'
    node_list.crs = crs_utm
    edge_list.crs = crs_utm
    for way in linesGPD.way.unique():
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,3))
        fig.suptitle('Tram Line: {}'.format(way), fontsize=12, fontweight='bold')
        ax.set_aspect("equal")
        stopsGPD[stopsGPD.way==way].plot(ax=ax, color = 'red')
        edge_list[edge_list.way==way].plot(ax=ax, color = 'grey')
        plt.show()
    
    #create network
    G = ox.gdfs_to_graph(node_list, edge_list)
    G = create_transfers(G, weight = 10)
    G.name = 'tram'
    return(G)

def find_nearest_node(data, nodes, spatial_index, buff = 50):
    """ 
    Given two networks find nearest nodes in target network for
    all nodes in source network by recursion
    
    parameters:
        tn : target network
        sn : source network
    returns:
        pandas.series: index of all records in tn that are nearest sn
    """
    polygon = data['geometry'].buffer(buff)
    possible_matches_index = list(spatial_index.intersection(polygon.bounds))
    possible_matches = nodes.iloc[possible_matches_index]
    if len(possible_matches) == 0:
        buff += 50
        v = find_nearest_node(data, nodes, spatial_index, buff)
    elif len(possible_matches) == 1:
        v = possible_matches.index[0]
        return v
    elif len(possible_matches) > 1:
        p1 = data['geometry']
        dist = possible_matches.distance(p1)
        v = dist[dist == min(dist)].index[0]
        return v
    else:
        print('no match')
    return v



    
def create_multiplex(street_network, transport_networks, waiting_times):
    """
    Create a multiplex based on different transport networks
    
    parameters:
        street_network: street graph (networkx graph object)
        transport_networks: dictionary containing transport network graphs (networkx graph object)
        waiting_times: dictionary containing waiting times between transport_networks and street_network
        
    returns:
        multiplex: networkx.MultiDiGraph
    """

    #set transport type attribute to nodes
    nx.set_node_attributes(street_network, 'network_type', street_network.name)
    nx.set_node_attributes(street_network, 'z', 0)
    #relabel nodes
    node_ori = [u for u in street_network.nodes(data=False)]
    node_new = ['{}-{}'.format(u, list(street_network.name)[0]) for u in node_ori]
    node_dict= dict(zip(node_ori, node_new))
    street_network = nx.relabel_nodes(street_network, node_dict)

    multiplex = street_network.copy()
    multiplex.name = 'multiplex'
    nx.set_edge_attributes(multiplex, 'transfer', False)

    i = 1
    for G in transport_networks:
        nx.set_node_attributes(G, 'network_type', G.name)
        nx.set_node_attributes(G, 'z', i)
        nx.set_edge_attributes(G, 'transfer', False)
        #relabel nodes
        node_ori = [u for u in G.nodes(data=False)]
        node_new = ['{}-{}'.format(u, list(G.name)[0]) for u in node_ori]
        node_dict= dict(zip(node_ori, node_new))
        G = nx.relabel_nodes(G, node_dict)

        #add node and edges to graph
        multiplex.add_nodes_from(G.nodes(data=True))
        multiplex.add_edges_from(G.edges(data=True))

        stops = pd.DataFrame.from_dict(G.node, orient = 'index')
        stops = gpd.GeoDataFrame(stops)

        nodes = pd.DataFrame.from_dict(street_network.node, orient = 'index')
        nodes = gpd.GeoDataFrame(nodes)
        spatial_index = nodes.sindex

        #find nearest street node to every stop to join network
        for u, data in G.nodes(data=True):
            v = find_nearest_node(data, nodes, spatial_index)
            # print('adding edge between: u = {} and v={}'.format(u,v))
            multiplex.add_edge(u = u,
                               v = v, 
                               transfer = True,
                               transfer_type = G.name,
                               weight = 0.1,
                              )
            multiplex.add_edge(u = v,
                   v = u, 
                   transfer = True,
                   transfer_type = G.name,
                   weight = waiting_times[i-1],
                  )
        i += 1

    nx.set_edge_attributes(multiplex, 'osmid', range(len(list(multiplex.edges()))))
    return multiplex
    

def plot_network(G, area):
    """
    Creates plot for network and urban area.

    Parameters
    ----------
	G: transport network as networkx multidigraph
	area: boundary as geopandas geoseries 

	Returns
    ----------
	None
    """

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

def plot_multiplex(multiplex, save = False):
    """
    Creates plot for network and urban area.

    Parameters
    ----------
    G: transport network as networkx multidigraph
    area: boundary as geopandas geoseries 

    Returns
    ----------
    None
    """

    G = nx.convert_node_labels_to_integers(multiplex)
    node_Xs = [float(node['x']) for node in G.node.values()]
    node_Ys = [float(node['y']) for node in G.node.values()]
    node_Zs = np.array([float(d['z'])*1000 for i, d in G.nodes(data=True)])
    node_size = []
    size = 1
    node_color =[]

    for i, d in G.nodes(data=True):
        if d['network_type']=='street':
            node_size.append(size)
            node_color.append('#66ccff')
        elif d['network_type']=='bus':
            node_size.append(size*4)
            node_color.append('#fb9a98')
        elif d['network_type']=='tram':
            node_size.append(size*8)
            node_color.append('#9bc37e') 

    lines = []
    lineWidth =[]
    lwidth = 0.2
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            xs, ys = data['geometry'].xy
            zs = [G.node[u]['z']*1000 for i in range(len(xs))] 
            lines.append([list(a) for a in zip(xs, ys, zs)])
            if data['transfer']:
                if data['transfer_type']=='bus':
                    lineWidth.append(lwidth/4)
                else:
                    lineWidth.append(lwidth/1.5)
            else:
                lineWidth.append(lwidth)
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.node[u]['x']
            y1 = G.node[u]['y']
            z1 = G.node[u]['z']*1000
            x2 = G.node[v]['x']
            y2 = G.node[v]['y']
            z2 = G.node[v]['z']*1000
            line = [[x1, y1, z1], [x2, y2, z2]]
            lines.append(line)
            if data['transfer']:
                if data['transfer_type']=='bus':
                    lineWidth.append(lwidth/6)
                else:
                    lineWidth.append(lwidth/1.5)
            else:
                lineWidth.append(lwidth)

    fig_height=15
    lc = Line3DCollection(lines, linewidths=lineWidth, alpha = 1, color = '#999999', zorder=1)
    edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
    west, south, east, north = edges.total_bounds
    bbox_aspect_ratio = (north-south)/(east-west)
    fig_width = fig_height / bbox_aspect_ratio
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.gca(projection='3d')
    ax.add_collection3d(lc)
    ax.scatter(node_Xs, node_Ys, node_Zs, s=node_size, c=node_color, zorder=2)

    ax.set_ylim(south, north)
    ax.set_xlim(west, east)
    ax.set_zlim(0, 2500)
    ax.axis('off')
    ax.margins(0)
    ax.tick_params(which='both', direction='in')
    fig.canvas.draw()
    # ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    plt.show()

    if save:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        filename = 'multiplex'
        output_img = 'images'
        file_format = 'png'
        path_filename = '{}/{}.{}'.format(output_img, filename, file_format)
        fig.savefig(path_filename, dpi=300, bbox_inches=extent, format=file_format, facecolor=fig.get_facecolor(), transparent=True)


def network_stats(G, area):
    """
    Calculates descriptive statistics for transport network
    Based On:
        shortest path betweenness 
        random walk betweenness
        reaching centrality
        information centrality
    
    G: networkx multidigraph 
    area: area in meters of urban extent
    
    returns: 
    G: multidigraph with added measures
    """
    area = area / 1e6
    
    #number of nodes and edges of graph
    n = len(list(G.nodes()))
    m = len(list(G.edges()))

    print('calculating shortest path betweenness...')
    eb = nx.edge_betweenness_centrality(G, weight = 'weight')
    nb = nx.betweenness_centrality(G, weight = 'weight')

    print('calculating random walk betweenness...')
    G_undirected = G.to_undirected(reciprocal=False)
    erwb = nx.edge_current_flow_betweenness_centrality(G_undirected, weight = 'weight')
    nrwb = nx.current_flow_betweenness_centrality(G_undirected, weight = 'weight')


    print('calculating reaching centrality...')
    # G_line = nx.line_graph(G)
    # erc = nx.local_reaching_centrality(G_line, G.nodes())
    # nrc = nx.local_reaching_centrality(G, G.nodes())

    print('calculating information centrality...')
    G_line_undirected = nx.line_graph(G_undirected)
    eic = nx.current_flow_closeness_centrality(G_line_undirected, weight = 'weight')
    nic = nx.current_flow_closeness_centrality(G_undirected, weight = 'weight')

    measures = [[eb,nb, G, 'Betweenness Centrality'], 
            [erwb,nrwb,G, 'Random Walk Betweenness'],
            # [erc,nrc,G, 'Reaching Centrality'],
            [eic,nic,G, 'Information Centrality']
           ]   

    print('generating plots...')
    ##plot values
    for measure in measures:
        print('calculating measures for: {}'.format(measure[3]))
        ev = []
        keys = []
        for (u,v,k) in measure[2].edges(keys=True):
            keys.append((u,v,k))
            if (u,v,k) in measure[0].keys():
                ev.append(measure[0][(u,v,k)])
            elif (u,v) in measure[0].keys():
                ev.append(measure[0][(u,v)])
            elif (u,v,k) in measure[0].keys():
                ev.append(measure[0][(v,u,k)])
            else:
                ev.append(-1)
        evdict = dict(zip(keys, ev))
        enorm = colors.Normalize(vmin=min(ev), vmax=max(ev))
        ecmap = cm.ScalarMappable(norm=enorm, cmap=cm.viridis)
        ec= [ecmap.to_rgba(cl) for cl in ev]
        
        nv = [measure[1][node] for node in measure[2]]
        nnorm = colors.Normalize(vmin=min(nv), vmax=max(nv))
        ncmap = cm.ScalarMappable(norm=nnorm, cmap=cm.viridis)    
        nc= [ncmap.to_rgba(cl) for cl in nv]
        nx.set_node_attributes(G, measure[3], measure[1])
        nx.set_edge_attributes(G, measure[3], evdict)

        
        fig, ax = ox.plot_graph(measure[2],
                                fig_height = 10,
                                node_size = 15,
                                node_color = nc,
                                node_zorder=2,
                                edge_color=ec,
                                edge_linewidth=0.1,
                                edge_alpha=1,
                                bgcolor = '#202020',
                                show = True,
                                close = True,
                                dpi=600,
                                equal_aspect=True,
                               )
        g = sns.distplot(nv, hist=False, kde_kws=dict(cumulative=True), 
                     axlabel= measure[3])
        g.figure.savefig("images/{}.pdf".format(measure[3]), format = 'pdf')
    return G

def intersection_voronoi(network, area, crs_utm, plot = True):
	"""
	Create Voronoi tesselation from x, y coordinates of network nodes

	network: street network as multidigraph
	area: urban boundary as geoseries
	crs_utm: coordinate system of area
	plot: bol if plots area shown

	returns: 
	geopandas dataframe
	"""
	#create vor tessellation from x, y coordinates of nodes
	print('Generating voronoi tesselation...')
	nodeX= [float(node['x']) for node in network.node.values()]
	nodeY = [float(node['y']) for node in network.node.values()]

	points = np.column_stack((nodeX,nodeY))
	vorTess = Voronoi(points)
	if plot:
		voronoi_plot_2d(vorTess,
		                show_vertices = False,
		                show_points = False
		               )
		plt.show()
	#create polygons from voronoi tessellation
	print('Generating geodataframe...')
	lines = [
	    geometry.LineString(vorTess.vertices[line])
	    for line in vorTess.ridge_vertices
	    if -1 not in line
	]

	#create shapely polygon and turn into geopandas dataframe
	vorPoly = list(polygonize(lines))
	vorPolyGPD = gpd.GeoDataFrame(geometry = vorPoly)
	vorPolyGPD.crs = crs_utm

	#create intersection with urban limit
	nodePolyGPD = gpd.overlay(vorPolyGPD, area, how='intersection')
	nodePolyGPD = nodePolyGPD[nodePolyGPD.is_valid]
	#plot and save polygons
	if plot:
		nodePolyGPD.plot(figsize = (15,10))
		plt.show()

	return nodePolyGPD

def area_overlay(source, target, crs_utm):
    """
    Calculates area overlay given to geometries and initial values
    
    source: geopandas dataframe of source values
    target: geopandas dataframe of target geometry
    
    returns: 
    geopandas dataframe
    """


    print('Calculating area overlay...')
    #create empty column
    target['Q1'] = np.nan
    target['Q2'] = np.nan
    target['Q3'] = np.nan
    target['Q4'] = np.nan
    target['ICV'] = np.nan
    target['nDwellings'] = np.nan
    target['nPeople'] = np.nan

    Q1values = []
    Q2values = []
    Q3values = []
    Q4values = []
    ICVvalues = []
    nDvalues = []
    nPvalues = []
    print('Calculating values for polygon...')
    for i in range(len(target)):
        #set initial values
        q1 = 0
        q2 = 0
        q3 = 0
        q4 = 0
        icv = 0
        ndwellings = 0
        npeople = 0
        count = 0
        
        boundary = target.geometry[i]

        #create spatial index and find geometry within polygon
        sindex = source.sindex
        matches_index = list(sindex.intersection(boundary.bounds))
        
        for j in matches_index:
            
            inters = overlay(source.iloc[[j]],target.iloc[[i]], how = 'intersection')
            if len(inters) is not 0:
                count += 1
                total_area = sum(source.iloc[[j]].area)
                target_area = sum(target.iloc[[i]].area)
                inter_area = sum(inters.geometry.area)
                inters_ratio = inter_area/total_area

                #number of people within area of intersection
                ndwelling_val = inters_ratio * source.iloc[[j]].nDwellings.values[0]
                npeople_val = inters_ratio * source.iloc[[j]].nPeople.values[0]
                
                #recalculate people belonging to each group
                q1_val = npeople_val * source.iloc[[j]].Q1.values[0]
                q2_val = npeople_val * source.iloc[[j]].Q2.values[0]
                q3_val = npeople_val * source.iloc[[j]].Q3.values[0]
                q4_val = npeople_val * source.iloc[[j]].Q4.values[0]
                
                #for icv value, get weighted mean
                icv_val = source.iloc[[j]].ICV.values[0]
                
            else:
                q1_val = 0
                q2_val = 0
                q3_val = 0
                q4_val = 0
                icv_val = 0
                ndwelling_val = 0
                npeople_val = 0
            q1 = q1 + q1_val
            q2 = q2 + q2_val
            q3 = q3 + q3_val
            q4 = q4 + q4_val
            icv = icv + icv_val
            ndwellings = ndwellings + ndwelling_val
            npeople = npeople + npeople_val
        
        if npeople != 0:
            Q1values.append(q1/npeople)
            Q2values.append(q2/npeople)
            Q3values.append(q3/npeople)
            Q4values.append(q4/npeople)
        else:
            Q1values.append(0)
            Q2values.append(0)
            Q3values.append(0)
            Q4values.append(0)
        if count != 0 :
            ICVvalues.append(icv/count)
        else:
            ICVvalues.append(np.nan)
        nDvalues.append(ndwellings)
        nPvalues.append(npeople)
    print('Appending values to geometry...')
    target['Q1'] = Q1values
    target['Q2'] = Q2values
    target['Q3'] = Q3values
    target['Q4'] = Q4values
    target['ICV'] = ICVvalues
    target['nDwellings'] = nDvalues
    target['nPeople'] = nPvalues
    res = gpd.GeoDataFrame(target)
    res.crs = crs_utm
    return res

def blocks_to_nodes(network, blocks, area, crs_utm, plot = True):
    """
    Maps population values from blocks to street intersections using 
    voronoi tesselation and weighted area overlay interpolation
    
    network: street network as multidigraph
    blocks: geopandas dataframe of source data
    area: urban boundary as geoseries
    crs_utm: coordinate system of area
    plot: bol if plots area shown
    
    returns: 
    networkx multidigraph
    """
    nodePolyGPD = intersection_voronoi(network, area, crs_utm, plot=plot)
    overlayPoly = area_overlay(blocks, nodePolyGPD, crs_utm)

    # assign values to nodes in street_graph using spatial join
    intersections = {node:data for node, data in network.nodes(data=True)}
    intersectionGPD = gpd.GeoDataFrame(intersections).T
    intersectionGPD.crs = crs_utm
    intersectionGPD['geometry'] = intersectionGPD.apply(lambda row: geometry.Point(row['x'], row['y']), axis=1)

    intersection_attrib = gpd.sjoin(intersectionGPD, overlayPoly, how='inner', op='within')
    
    #assign values to graph
    nx.set_node_attributes(network, 'Q1', intersection_attrib['Q1'].to_dict())
    nx.set_node_attributes(network, 'Q2', intersection_attrib['Q2'].to_dict())
    nx.set_node_attributes(network, 'Q3', intersection_attrib['Q3'].to_dict())
    nx.set_node_attributes(network, 'Q4', intersection_attrib['Q4'].to_dict())
    nx.set_node_attributes(network, 'ICV', intersection_attrib['ICV'].to_dict())
    nx.set_node_attributes(network, 'nDwellings', intersection_attrib['nDwellings'].to_dict())
    nx.set_node_attributes(network, 'nPeople', intersection_attrib['nPeople'].to_dict())
    print('done!')
    return network


def rw_laplacian(G, weight = 'weight'):
    ##create adjecency matrix
    M = nx.to_scipy_sparse_matrix(G, 
                                  nodelist=nodelist, 
                                  weight=weight,
                                  dtype=float)
    n, m = M.shape

    #create diagnal
    DI = spdiags(1.0 / sp.array(M.sum(axis=1).flat), [0], n, n)

    #probability transition matrix
    P = DI * M

    #get eigenvectors and eigenvalues of probability matrix
    evals, evecs = linalg.eigs(P.T, k=1)

    v = evecs.flatten().real
    p = v / v.sum()
    sqrtp = sp.sqrt(p)

    Q = spdiags(sqrtp, [0], n, n) * P * spdiags(1.0 / sqrtp, [0], n, n)
    I = sp.identity(len(G))

    #directed laplacian matrix
    rwlaplacian = I - (Q + Q.T) / 2.0
    return rwlaplacian

def spatial_outreach(G, t=20):
    """
    Calculates spatial outreach (strano 2015) of street nodes in the mUltiplex
    
    G: multiplex
    T: travel cost
    
    returns: 
    dict with node ids and values
    """

    #get node list
    full_start_time = time.time()
    nodes = list(G.nodes())
    #get only street nodes
    street_nodes = ([s for s in nodes if 's' in s])
    soi = []
    for i in street_nodes:
        start_time = time.time()
        dist = []
        #get nodes within temporal constraint
        G_i = nx.ego_graph(G, i, radius = t, center=True, distance = 'weight')

        #get calculate euclidian distance
        nodes = nx.get_node_attributes(G_i,'geometry')
        ori_point = nodes[i]
        nodes.pop(i)
        dest_points = [nodes[k] for k in nodes.keys()]
        dist = np.array([ori_point.distance(dest) for dest in dest_points])
        avg_dist = dist.sum()/len(dist)
        soi.append(avg_dist)
        # print('calculated spatial outreach for node: {} in {:.2f} seconds, max: {}'.format(i,time.time() - start_time, max_dist))

    spatial_outreach_dict = dict(zip(street_nodes, soi))
    print('calculated spatial outreach for graph in {:.2f} seconds'.format(time.time() - full_start_time))
    return spatial_outreach_dict


def random__walk_segregation(G, group = "Q1", alpha = 0.85):
    ## get initial values as column vectors
    n_i = np.array(list(nx.get_node_attributes(G, 'nPeople').values()))
    n = n_i.sum()

    c_gi = np.array(list(nx.get_node_attributes(G,group).values()))
    n_gi = c_gi * n_i
    n_g = n_gi.sum()
    d_gi = n_gi / n_g

    c_gi.shape = (len(c_gi),1)
    d_gi.shape = (len(d_gi),1)

    # qij = probability of a journey starting in i ending in j
    # qij can we calculate this with a temporal constraint? using spatial outreach
    #get weighted adj. matrix
    W = nx.adjacency_matrix(G, weight='betweenness')
    #add self-edge in matrix and invert distance - higher values if closer distance

    #Get degree matrix
    Degree = sp.sparse.spdiags(1./W.sum(1).T, 0, *W.shape)

    #create identity matrix
    I = sp.identity(len(G))

    #create row stochastic matrix
    P = Degree * W

    #create matrix Q such that q_ij is the probability that walk started in i ends in j
    Q = (1-alpha) * (I - alpha * P).I * P

    #calculate isolation index
    isolation_gi = np.multiply(den_gi,(Q*con_gi))

    norm_isolation_gi = (n_g/n)**-1 * isolation_gi
    sigma_bar = list(np.array(norm_isolation_gi.flatten())[0])

    res = dict(zip(list(G.nodes()), sigma_bar)) 

    return res

def multiplex_stats(G):
    """
    Calculates descriptive statistics for multiplex
    Based On:
        shortest path betweenness 
        spatial outreach
    
    G: networkx multidigraph 
    area: area in meters of urban extent
    
    returns: 
    G: multidigraph with added measures
    """
    
    #number of nodes and edges of graph
    n = len(list(G.nodes()))
    m = len(list(G.edges()))

    print('calculating shortest path betweenness...')
    eb = nx.edge_betweenness_centrality(G, weight = 'weight')
    nb = nx.betweenness_centrality(G, weight = 'weight')

    print('calculating spatial outreach...')
    soi = spatial_outreach(G, t=20)

    measures = [[eb,nb, G, 'Betweenness Centrality'], 
               [soi,0,G, 'Spatial Outreach']
           ] 

    print('generating plots...')
    ##plot values
    for measure in measures:
        print('calculating measures for: {}'.format(measure[3]))
        ev = []
        keys = []
        if len(measure[0]) != 0:
            for (u,v,k) in measure[2].edges(keys=True):
                keys.append((u,v,k))
                if (u,v,k) in measure[0].keys():
                    ev.append(measure[0][(u,v,k)])
                elif (u,v) in measure[0].keys():
                    ev.append(measure[0][(u,v)])
                elif (u,v,k) in measure[0].keys():
                    ev.append(measure[0][(v,u,k)])
                else:
                    ev.append(-1)
        else:
            for (u,v,k) in measure[2].edges(keys=True):
                keys.append((u,v,k))
                ev.append(0)

        evdict = dict(zip(keys, ev))
        enorm = colors.Normalize(vmin=min(ev), vmax=max(ev))
        ecmap = cm.ScalarMappable(norm=enorm, cmap=cm.viridis)
        ec= [ecmap.to_rgba(cl) for cl in ev]
        
        nv = [measure[1][node] for node in measure[2]]
        nnorm = colors.Normalize(vmin=min(nv), vmax=max(nv))
        ncmap = cm.ScalarMappable(norm=nnorm, cmap=cm.viridis)    
        nc= [ncmap.to_rgba(cl) for cl in nv]
        nx.set_node_attributes(G, measure[3], measure[1])
        nx.set_edge_attributes(G, measure[3], evdict)

        
        fig, ax = ox.plot_graph(measure[2],
                                fig_height = 10,
                                node_size = 15,
                                node_color = nc,
                                node_zorder=2,
                                edge_color=ec,
                                edge_linewidth=0.1,
                                edge_alpha=1,
                                bgcolor = '#202020',
                                show = True,
                                close = True,
                                dpi=600,
                                equal_aspect=True,
                               )
        g = sns.distplot(nv, hist=False, kde_kws=dict(cumulative=True), 
                     axlabel= measure[3])
        g.figure.savefig("images/{}.pdf".format(measure[3]), format = 'pdf')
    return G

