#%% Startup
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:46:29 2022

@author: L.J. Koenders, 20172832
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
from geographiclib.geodesic import Geodesic
# from geographiclib.geodesic2 import Geodesic as Geodesic2
# from geographiclib.geodesicFast import Geodesic as GeodesicFast
GD = Geodesic.WGS84 # original
# GD2= Geodesic2.WGS84 # latest properly working GDF version
# GDF= GeodesicFast.WGS84 # stripped version for quick Inverse() calculations

import copy

#%% Graph manipulation
def AddEdges(graph, B, E, length=[], speed=[]):
    """
    Add edges to the graph in various ways. B is begin node, E is end node.
    
    Parameters
    ----------
    graph : MultiDiGraph
        MultiDiGraph object to add edges to
    B, E : int, list
        Begin and End node(s) respectively for edge(s) to add.
        1. B, E : int, int
        2. [B,...], [E,...] : list[int], list[int]
    length : float, list, optional
        Lenght of edge(s) to add. Default = None.
        1. length : float
        2. length : list[float]
    """

    if type(B) != type(E):
        raise Exception("B and E should be of the same type.")
    elif type(B) == int:
        graph = __AddSingleEdge(graph, B, E, length, speed)
    elif type(B) == list:
        graph = __AddMultipleEdge(graph, B, E, length, speed)
    else:
        raise Exception("Unknown input type.")
    
    return graph

def AddNodeWithEdge(graph, startNode, length, angle, unit="rad", nodeName=None):
    """
    Add a node to a network by specifying an outgoing edge from an existing node.
    
    Parameters
    ----------
    startNode : int
        Specify the node where the new edge will start.
    length : int, float
        Specify the length of the new edge.
    angle : int, float
        Specify the angle of the new edge.
    unit : "rad" (default), "deg", optional
        Specify the unit of the given angle.
    nodeName : str, optional
        Specify new node's name.
    """
    currentNrNodes = len(list(graph._node.keys()))
    
    if startNode not in graph.nodes:
        raise Exception(f"Node {startNode} does not exist.")
    elif length <= 0:
        raise Exception("Length must be larger than 0.")
    else:
        if nodeName == None:
            nodeName = [currentNrNodes]
        elif type(nodeName) != list:
            nodeName = list(nodeName)
            
        if unit == "rad":
            relX = np.cos(angle)*length
            relY = np.sin(angle)*length
        elif unit == "deg":
            relX = np.cos(angle/180*np.pi)*length
            relY = np.sin(angle/180*np.pi)*length
        else:
            raise Exception("Unit must be either rad or deg.")
        X = relX + graph._node[startNode]['x']
        Y = relY + graph._node[startNode]['y']
        graph = __AddSingleNode(graph, nodeName, X, Y)
        graph = __AddSingleEdge(graph, startNode, nodeName[0])
    return graph

def NewGraph():
    return nx.DiGraph()

def SplitArcWithNode(graph, startNode, endNode, length, unit="abs"):
    """
    Split an arc by adding a node in between.
    
    Parameters
    ----------
    startNode : int
        Specify the node where the new edge will start.
    endNode : int
        Specify the direction of the new edge with an endnode.
    length : int, float
        Specify the length of the new edge.
    unit : "abs" (default), "rel", optional
        Absolute or relative value of edge length. Absolute in meters, relative in fraction of edge B-E.
    """
    currentNodes = list(graph._node.keys())
    
    if startNode not in currentNodes:
        raise Exception(f"Node {startNode} does not exist.")
    elif endNode not in currentNodes:
        raise Exception(f"Node {endNode} does not exist.")
    elif length <= 0:
        raise Exception("Length must be larger than 0.")
    elif unit not in ["abs","rel"]:
        raise Exception("Unit must be either abs or rel.")
    else:
        # Remove original arc from B to E
        graph = RemoveEdge(graph, startNode, endNode)
        
        # Calculate coordinates of "split-up node"
        Xb = graph._node[startNode]['x']
        Yb = graph._node[startNode]['y']
        Xe = graph._node[endNode]['x']
        Ye = graph._node[endNode]['y']
        geoLine = GD.InverseLine(Yb, Xb, Ye, Xe)
        
        if unit == "rel":
            length = length*geoLine.s13
        
        newCoords = geoLine.Position(length)
        Xnew = round(newCoords['lon2'], 6)
        Ynew = round(newCoords['lat2'], 6)
        
        # Add "split-up node" to graph
        coordList = GetCoordList(graph)
        if [Xnew, Ynew] in coordList.values(): # node already exists, reuse it
            splitNodeNr = list(coordList.keys())[list(coordList.values()).index([Xnew,Ynew])]
        else:
            splitNodeNr = max(currentNodes) + 1 # take new node number
            graph = __AddSingleNode(graph, [splitNodeNr], Xnew, Ynew)
        
        # Connect new node with original nodes
        graph = __AddSingleEdge(graph, startNode, splitNodeNr, length=None, speed=30)
        graph = __AddSingleEdge(graph, splitNodeNr, endNode, length=None, speed=30)
    return graph

def AddNodes(graph, X, Y, nodeList=None):
    """
    Add node(s) to the graph in various ways.
        1. X, Y : float, float
        2. [X,...], [Y,...] : list[float], list[float]
    """
    if "networkx.classes" not in str(type(graph)):
        graph = NewGraph()
    
    currentNrNodes = len(list(graph._node.keys()))
    
    if type(X) != type(Y):
        raise Exception("X and Y should be of the same type.")
    elif type(X) == int or type(X) == float:
        if nodeList == None:
            nodeList = [currentNrNodes]
        graph = __AddSingleNode(graph, nodeList, X, Y)
    elif type(X) == list:
        if nodeList == None:
            nodeList = list(range(currentNrNodes, currentNrNodes + len(X)))
        graph = __AddMultipleNode(graph, nodeList, X, Y)
    else:
        raise Exception("Unknown input type.")
    
    return graph

def ChangeEdgeLength(graphOld, B:int, E:int, nodeMove:str, length:float, method:str="geo", deepcopy=False, debug=-1):
    """
    Change length of an edge from a graph to specified length.
    
    Parameters
    ----------
    graph : graph object
        Graph of edge.
    B, E : int
        Begin and end nodeNr respectively of edge to change.
    nodeMove : str
        Node to move position. "B", "E" or "mid"
    length : float
        New length of edge, in meters.
    method : str
        "geo", length is in geodesical meters. "graph", length is in coordinate meters.
    """
    if deepcopy: graph = copy.deepcopy(graphOld)
    else:        graph= graphOld
    
    currentNodes = list(graph._node.keys())
    if B not in currentNodes:
        raise Exception(f"Node {B} does not exist.")
    elif E not in currentNodes:
        raise Exception(f"Node {E} does not exist.")
    elif graph.has_edge(B, E) == False:
        raise Exception(f"Edge between {B} and {E}  does not exist.")
    else:
        Xb = graph._node[B]['x']
        Yb = graph._node[B]['y']
        Xe = graph._node[E]['x']
        Ye = graph._node[E]['y']
        
        if method == "geo":
            oldLength = graph[B][E]["length"]
            lengthFraction = length/oldLength
        else:
            raise Warning("This might not work as intended.")
            Xold = Xe - Xb
            Yold = Ye - Yb
            angle = np.angle(Xold + Yold*1j) # use complex number to get full 2pi rad range
            
            relX  = np.cos(angle)*length
            relY  = np.sin(angle)*length
        
        if nodeMove == "B":
            if method == "geo":
                geoLine = GD.InverseLine(Yb, Xb, Ye, Xe)
                Bfrac = 1-lengthFraction
                
                newB = geoLine.Position(Bfrac * geoLine.s13)
                Xnew = round(newB['lon2'], 6)
                Ynew = round(newB['lat2'], 6)
            else:
                Xnew = Xe + relX
                Ynew = Ye + relY
            graph = ChangeNode(graph, B, "coordsNew", (Xnew, Ynew))
        elif nodeMove == "E":
            if method == "geo":
                geoLine = GD.InverseLine(Ye, Xe, Yb, Xb)
                Efrac = 1-lengthFraction
                
                newE = geoLine.Position(Efrac * geoLine.s13)
                Xnew = round(newE['lon2'], 6)
                Ynew = round(newE['lat2'], 6)
            else:
                Xnew = Xb + relX
                Ynew = Yb + relY
            graph = ChangeNode(graph, E, "coordsNew", (Xnew, Ynew))
        elif nodeMove == "mid":
            if method == "geo":
                geoLine = GD.InverseLine(Yb, Xb, Ye, Xe)
                Bfrac = (1-lengthFraction)/2
                Efrac = Bfrac + lengthFraction
                
                newB = geoLine.Position(Bfrac * geoLine.s13)
                # Xbnew = round(newB['lon2'], 6)
                # Ybnew = round(newB['lat2'], 6)
                Xbnew = newB['lon2']
                Ybnew = newB['lat2']
                newE = geoLine.Position(Efrac * geoLine.s13)
                # Xenew = round(newE['lon2'], 6)
                # Yenew = round(newE['lat2'], 6)
                Xenew = newE['lon2']
                Yenew = newE['lat2']
                
                if debug != -1:
                    before = graph[B][E]['length']

                graph = ChangeNode(graph, B, "coordsNew", (Xbnew, Ybnew)) # (lon,lat) cuz (X,Y)
                graph = ChangeNode(graph, E, "coordsNew", (Xenew, Yenew)) # (lon,lat) cuz (X,Y)
                
                if debug != -1 and False:
                    print(f" Iteration {debug[0], debug[1]}")
                    print(f" Before: {round(before,3)}")
                    print(f" After:  {round(graph[B][E]['length'],5)} (goal: {round(length,5)})")
            else:
                raise Exception("Who uses this? Just use the geodesic method.")
            
    return graph

def ChangeNode(graph, nodeNr, changeType, *args):
    """
    Change the data of a node in a graph, for changeType =
        1. "coordsNew", (X,Y) : change the coordinates of a node using a new tuple.
        2. "coordsAdd", (X,Y) : add to original coordinates of a node.
        3. "remove" : remove nodeNr from the graph.
    """
    
    # Option 1
    if changeType == "coordsNew":
        if len(args) == 1 and type(args[0]) == tuple:
            if nodeNr in graph._node.keys():
                graph._node[nodeNr]['x'] = args[0][0]
                graph._node[nodeNr]['y'] = args[0][1]
                graph = __ChangeAdj(graph, nodeNr)
            else:
                raise Exception(f"Node {nodeNr} does not exist.")
        else:
            raise Exception("Coordinates should be of type tuple.")
    # Option 2
    elif changeType == "coordsAdd":
        if len(args) == 1 and type(args[0]) == tuple:
            if nodeNr in graph._node.keys():
                graph._node[nodeNr]['x'] += args[0][0]
                graph._node[nodeNr]['y'] += args[0][1]
                graph = __ChangeAdj(graph, nodeNr)
            else:
                raise Exception(f"Node {nodeNr} does not exist.")
        else:
            raise Exception("Coordinates should be of type tuple.")
    # Option 3
    elif changeType == "remove":
        if len(args) == 0:
            graph.remove_nodes_from([nodeNr])
        else:
            raise Exception("Change type \"remove\" does not take any arguments.")
            
    else:
        raise Exception(f"Change type \"{changeType}\" does not exist.")
    return graph

def GetNodeCoords(graph, nodeNr):
    """
    Get coordinates from a node in the graph.
    
    Parameters
    ----------
    nodeNr : int
        Node to get coordinates for.
    """
    currentNrNodes = list(graph._node.keys())
    if nodeNr not in currentNrNodes:
        raise Exception(f"Node {nodeNr} does not exist.")
        
    X = graph._node[nodeNr]['x']
    Y = graph._node[nodeNr]['y']
    return (X, Y)

def MergeNodes(graph, node1, node2):
    # Collect all arcs to and from both nodes
    incoming = list(graph.predecessors(node1)) + list(graph.predecessors(node2))
    outgoing = list(graph.successors(node1))   + list(graph.successors(node2))
    
    # Make new "merge node"
    currentNodes = list(graph._node.keys())
    newNodeNr = max(currentNodes) + 1
    
    Xb = graph._node[node1]['x']
    Yb = graph._node[node1]['y']
    Xe = graph._node[node2]['x']
    Ye = graph._node[node2]['y']
    geoLine = GD.InverseLine(Yb, Xb, Ye, Xe)
    
    newCoords = geoLine.Position(0.5*geoLine.s13) # in the middle of both nodes
    Xnew = round(newCoords['lon2'], 6)
    Ynew = round(newCoords['lat2'], 6)
    
    # Remove old nodes and add "merge node" to graph
    graph = ChangeNode(graph, node1, "remove")
    graph = ChangeNode(graph, node2, "remove")
    graph = __AddSingleNode(graph, [newNodeNr], Xnew, Ynew)
    
    # Add all edges back
    for B in incoming:
        if B not in [node1, node2]:
            graph = __AddSingleEdge(graph, B, newNodeNr, length=None, speed=30)
    for E in outgoing:
        if E not in [node1, node2]:
            graph = __AddSingleEdge(graph, newNodeNr, E, length=None, speed=30)
    
    return graph

def RemoveEdge(graph, B, E):
    """
    Remove an edge from a graph.
    
    Parameters
    ----------
    B : int
        Begin node of edge.
    E : int
        End node of edge.
    """
    currentNrNodes = list(graph._node.keys())
    if B not in currentNrNodes:
        raise Exception(f"Node {B} does not exist.")
    elif E not in currentNrNodes:
        raise Exception(f"Node {E} does not exist.")
    elif graph.has_edge(B, E) == False:
        raise Exception(f"Edge between {B} and {E}  does not exist.")
    else:
        graph.remove_edge(B, E)
    return graph

def __AddMultipleEdge(graph, B, E, length=[], speed=[]):
    if length == [] and speed == []:
        for idx in range(len(B)):
            graph.add_edge(B[idx], E[idx])
            graph = __ChangeAdj(graph, B[idx], E[idx])
    elif length != [] and speed == []:
        for idx in range(len(B)):
            graph.add_edge(B[idx], E[idx], length=length[idx])
            graph = __ChangeAdj(graph, B[idx], E[idx], length=length[idx])
    elif length == [] and speed != []:
        for idx in range(len(B)):
            graph.add_edge(B[idx], E[idx], speed=speed[idx])
            graph = __ChangeAdj(graph, B[idx], E[idx], speed=speed[idx])
    elif length != [] and speed != []:
        for idx in range(len(B)):
            graph.add_edge(B[idx], E[idx], length=length[idx], speed=speed[idx])
            graph = __ChangeAdj(graph, B[idx], E[idx], length=length[idx], speed=speed[idx])
    return graph

def __AddMultipleNode(graph, nodeList, X, Y):
    for idx in range(len(X)):
        graph.add_nodes_from([nodeList[idx]], x=X[idx], y=Y[idx])
    return graph

def __AddSingleEdge(graph, B, E, length=None, speed=None):
    if length == None and speed == None:
        graph.add_edge(B, E)
        graph = __ChangeAdj(graph, B, E)
    elif length != None and speed == None:
        graph.add_edge(B, E, length=length)
        graph = __ChangeAdj(graph, B, E, length=length)
    elif length == None and speed != None:
        graph.add_edge(B, E, speed=speed)
        graph = __ChangeAdj(graph, B, E, speed=speed)
    elif length != None and speed != None:
        graph.add_edge(B, E, length=length, speed=speed)
        graph = __ChangeAdj(graph, B, E, length=length, speed=speed)
    return graph

def __AddSingleNode(graph, nodeList, X, Y):
    graph.add_nodes_from([nodeList[0]], x=X, y=Y)
    return graph
  
def __ChangeAdj(graph, beginNode, endNode=None, length=None, speed=None):
    edgeList = []
    if endNode != None: # if a specific edge is specified
        edgeList += [(beginNode, endNode)]
    else: # if all edges of a node have to be changed
        if type(graph) == type(nx.DiGraph()):
            incoming = list(graph.predecessors(beginNode))
            outgoing = list(graph.successors(beginNode))
            for node in incoming:
                edgeList += [(node, beginNode)]
            for node in outgoing:
                edgeList += [(beginNode, node)]
        elif type(graph) == type(nx.Graph()):
            connected = list(graph.neighbors(beginNode))
            for node in connected:
                edgeList += [(node, beginNode)]
        else:
            raise Exception(f"Graph type {type(graph)} not supported. Use {type(nx.Graph)} or {type(nx.DiGraph)} instead.")
    
    for edge in edgeList:
        B = edge[0]
        E = edge[1]
        
        if length == None: # length is not given, so calculate it
            # Calculate length between nodes
            coordBegin = (graph._node[B]['x'], graph._node[B]['y'])
            coordEnd   = (graph._node[E]['x'], graph._node[E]['y'])
            lengthNew = GD.Inverse(coordBegin[1], coordBegin[0], coordEnd[1], coordEnd[0])["s12"]
            # lengthNew  = GDF.Inverse(coordBegin[1], coordBegin[0], coordEnd[1], coordEnd[0]) # GDF got lost when updating Spyder
            # lengthNew2 = GD.Inverse(coordBegin[1], coordBegin[0], coordEnd[1], coordEnd[0])["s12"]
            # if lengthNew != lengthNew2:
            #     raise Exception(f"Fast distance calculation not correct! \nlengthNew1: {lengthNew}\nlengthNew2: {lengthNew2}")
            graph._adj[B][E]['length'] = lengthNew
        else:
            graph._adj[B][E]['length'] = length
            
        if speed != None:
            graph._adj[B][E]['speed'] = speed
    
    return graph

#%% Graph information
def FindClosestNodeInfo(coordList, coordsOrNode, method="geodesic", graph=None, initNode=None):
    """
    Find the closest node in a coordList, starting from a given coordinate. Optionally following paths of a given graph.
    Returns [nodeName, distance]
    
    Parameters
    ----------
    coordList : dict
        The dictionary of nodes and their coordinates.
    coordsOrNode : list, tuple, int
        Method="geodesic": The coordinate [X,Y] of the point you want to find.
        Method="path": The node number of the point you want to find.
    method : string, optional
        Method of searching, option for "geodesic" or "path" search. default = "geodesic".
    graph : graph, optional
        To perform a path search, the graph object is required.
    initNode : int
        An initial guess of the closest node in coordList can be given, to potentially skip many distance calculations.
    """
    if initNode != None: # if initial node is given, start with its distance
        initDistance = GD.Inverse(coordsOrNode[1], coordsOrNode[0], coordList[initNode][1], coordList[initNode][0])["s12"]
        # initDistance = GDF.Inverse(coordsOrNode[1], coordsOrNode[0], coordList[initNode][1], coordList[initNode][0]) # GDF got lost when updating Spyder
        closestNode = [initNode, initDistance]
    else:
        closestNode = [None, np.Inf]
    
    for node in coordList:
        if method=="geodesic":
            coordsNode = coordList[node]
            
            # Calculate difference in X and Y coordinate
            if closestNode[0] != None:
                # Calculate radius of initial guess
                try: coordsClose = coordList[closestNode[0]]
                except Exception as e:
                    print(e)
                    print(closestNode)
                coordsClose = coordList[closestNode[0]]
                diffCX = coordsClose[0]-coordsOrNode[0]
                diffCY = coordsClose[1]-coordsOrNode[1]
                diffNX =  coordsNode[0]-coordsOrNode[0]
                diffNY =  coordsNode[1]-coordsOrNode[1]
            else:
                # No guess (yet), always check
                diffCX = diffCY = np.Inf
                diffNX = diffNY = 0
            
            # If new coordinate is outside current radius, there's not point in checking it
            radiusC = (diffCX**2+diffCY**2)**0.5
            radiusN = (diffNX**2+diffNY**2)**0.5
            
            if radiusN < max(radiusC*1.1, 0.01): # 10% safety margin, but minimum of 0.01 (for small distances)
                # distance = GDF.Inverse(coordsOrNode[1], coordsOrNode[0], coordsNode[1], coordsNode[0]) # GDF got lost when updating Spyder
                distance = GD.Inverse(coordsOrNode[1], coordsOrNode[0], coordsNode[1], coordsNode[0])["s12"]
            else:
                distance = np.Inf # don't try if direct distance is already longer
        elif method=="path":
            try:
                distance = nx.shortest_path_length(graph, source=coordsOrNode, target=node, weight="length")
            except:
                distance = np.Inf
            
        if distance < closestNode[1]:
            closestNode = [node, distance]
        
    return closestNode

def GetAdjacencyMatrix(graph):
    # graph = __ConvertFGToDiGraph(graph) # Take care of possible FlexibleGraph object type
    adjM = nx.linalg.graphmatrix.adjacency_matrix(graph)
    return adjM

def GetCoordList(graph):
    """
    Collect the coordinates of all nodes in a graph, type dict().
    Supported types: FlexibleGraph, DiGraph.
    """
    # graph = __ConvertFGToDiGraph(graph) # Take care of possible FlexibleGraph object type
    
    coordList = {}
    for idx in graph._node.keys():
        coordList[idx] = [graph._node[idx]['x'], graph._node[idx]['y']]
    return coordList

def GetEdgeList(graph):
    """
    Collect the edges in a graph, type list().
    Supported types: FlexibleGraph, DiGraph.
    """
    # graph = __ConvertFGToDiGraph(graph) # Take care of possible FlexibleGraph object type
    edgeList = []
    for B in graph._adj.keys():
        for E in graph._adj[B].keys():
            edgeList +=[(B,E)]
    return edgeList
    # return list(graph.edges)

def GetEdgeAttributeDict(graph, attribute):
    """
    Get a dictionary of all edges with the specified attribute.
    """
    edgeAttr = dict(graph.edges)
    for edge in edgeAttr:
        edgeAttr[edge] = graph[edge[0]][edge[1]][attribute]
    return edgeAttr

def GetNodeList(graph):
    """
    Collect the nodes in a graph, type list().
    Supported types: FlexibleGraph, DiGraph.
    """
    # graph = __ConvertFGToDiGraph(graph) # Take care of possible FlexibleGraph object type
    return list(graph._node.keys())

def GetGraphMap(fetchType, showBool:bool = False, saveBool:bool = False, DPI:int = 400):
    """
    Create a graph object using a bounding box or city name. Returns a graph object.
    
    Parameters
    ----------
    fetchType : list, str
        For bbox, use [north, east, south, west]. For place, use "City, Country".
    showBool : bool, optional (default = False)
        True to show plot.
    saveBool : bool, optional (default = False)
        True to save file.
    DPI : int, optional (default = 400)
        Change plot DPI.
    """
    if type(fetchType) == list and len(fetchType) == 4: # bbox
        north, east, south, west = fetchType[0], fetchType[1], fetchType[2], fetchType[3]
        graph = ox.graph_from_bbox(north, south, east, west, network_type = 'drive')
        if showBool:
            ox.plot_graph(graph, dpi=DPI, show=showBool, save=saveBool)
    elif type(fetchType) == str: # place
        graph = ox.graph_from_place(fetchType, network_type = 'drive')
        if showBool:
            ox.plot_graph(graph, dpi=DPI, show=showBool, save=saveBool, node_size=0.5, edge_linewidth=0.5)
    else:
        raise Exception("Only 4-long list (bbox) or string (place) types supported.")
    
    return graph

#%% Clustering
class ClusterGraphs:
    """
    Cluster nodes from a graph around given parent nodes.
    
    Parameters
    ----------
    graphToCluster : DiGraph
        Graph that has to be clustered.
    parentNodeDict : dict
        Dictionary of parent nodes.
        For method="geodesic", a dictionary with key:coordinate suffices.
        For method="path", a dictionary with minimally the structure {[closestID]:..., [coords]:...} is required.
    parentGuessDict : dict
        Dictionary of (for now) specific format with initial guesses for clustering.
    nodesToCluster : list (default = "all")
        Nodes that have to be clustered. This must be a subset of the nodes of graphToCluster.
    method : string (default = "geodesic")
        Cluster method: either "geodesic" (absolute distance) or "path" (follow graph edges).
    plot : bool (default = False)
    
    Returns
    -------
    Object with accessible properties:
        
    nodeInfo : dict
        Cluster information per node.
    parentCluster : dict
        List of child nodes per parent node.
    """
    def __init__(self, graphToCluster, parentNodeDict, parentGuessDict=None, nodesToCluster="all", method="geodesic", plot=False, plotAspect=1):
        # Get nodes to cluster
        if nodesToCluster == "all":
            nodesToCluster = list(graphToCluster._node)

        # Cluster nodes
        parentCoords = self.__Cluster(graphToCluster, parentNodeDict, parentGuessDict, nodesToCluster, method)

        # Plotting
        if plot:
            coordList = GetCoordList(graphToCluster)
            closestNodes = list(parentNodeDict["closestID"].values())
            closestNodesCoords = {k:coordList[k] for k in closestNodes}
            self.__Plotting(graphToCluster, parentCoords, closestNodesCoords, method, plotAspect)
    
    def __Cluster(self, graphToCluster, parentNodeDict, parentGuessDict, nodesToCluster, method):
        # Get right structure for parentCluster and parentCoords
        graphCoords = GetCoordList(graphToCluster)
        self.nodeInfo = dict()
        if method == "geodesic":
            try:
                self.parentCluster = {k:[] for k in parentNodeDict["coords"].keys()} # copy keys but not values from parentNodeDict
                parentCoords = {k:parentNodeDict["coords"][k] for k in parentNodeDict["coords"].keys()}
            except:
                self.parentCluster = {k:[] for k in parentNodeDict.keys()} # copy keys but not values from parentNodeDict
                parentCoords = parentNodeDict
        elif method == "path":
            self.parentCluster = {parentNodeDict["closestID"][k]:[] for k in parentNodeDict["coords"].keys()} # copy keys but not values from parentNodeDict
            parentCoords = {parentNodeDict["closestID"][k]:graphCoords[parentNodeDict["closestID"][k]] for k in parentNodeDict["coords"].keys()}
        
        # Clustering
        nodesClustered = set()
        functionVisited = 0
        for node in nodesToCluster:
            if node not in nodesClustered:
                coords = graphCoords[node]
                if method == "geodesic":
                    if parentGuessDict == None:
                        closestNode = FindClosestNodeInfo(parentCoords, coords)
                    else:
                        guessNode = [parentGuessDict[node]["closeParAbs"], parentNodeDict[node]["distParAbs"]]
                        closestNode = FindClosestNodeInfo(parentCoords, coords, initNode=guessNode)
                    functionVisited += 1
                    self.nodeInfo[node] = {"closeParAbs" : closestNode[0], "distParAbs" : closestNode[1]}
                    self.parentCluster[closestNode[0]] += [node]
                elif method == "path":
                    closestNode = FindClosestNodeInfo(parentCoords, node, method="path", graph=graphToCluster)
                    functionVisited += 1
                    try:
                        closestToParent = list(parentNodeDict["closestID"].keys())[list(parentNodeDict["closestID"].values()).index(closestNode[0])]
                        self.nodeInfo[node] = {"closeParRel" : closestToParent, "distParRelPath" : closestNode[1]}
                        
                        # Shortcut (append nodes on shortest path to same parent)
                        path = nx.shortest_path(graphToCluster, node, closestNode[0], weight="length")
                        for pathNode in path:
                            if pathNode not in nodesClustered:
                                distance = nx.shortest_path_length(graphToCluster, source=pathNode, target=closestNode[0], weight="length")
                                self.nodeInfo[pathNode] = {"closeParRel" : closestToParent, "distParRelPath" : distance}
                                self.parentCluster[closestNode[0]] += [pathNode]
                                nodesClustered |= {pathNode}
                    except: None
        # print(f"Computations saved: {len(nodesToCluster)-functionVisited}/{len(nodesToCluster)}")
        return parentCoords
    
    def __Plotting(self, graphToCluster, parentCoords, closestNodesCoords, method, plotAspect):
        if method == "geodesic": title = "Geodesic cluster"
        elif method == "path":   title = "Path cluster"
        PlotNewFigure(DPI=800, title=title, aspect=plotAspect)
        
        # Base graph
        PlotGraph(graphToCluster, arrowSize=0.01, nodeColour="black", nodeAlpha=0.25, nodeSize=5, nodeLabels=False, edgeWidth=0.25)
        
        # Clusters
        for idx, parent in enumerate(self.parentCluster):
            colour = f"C{idx}" # choose a new colour each time
            nx.draw_networkx_nodes(graphToCluster, GetCoordList(graphToCluster), nodelist=self.parentCluster[parent], node_size=10, node_color=colour, alpha=0.5)
        
        # Parent nodes
        nx.draw_networkx_nodes(graphToCluster, closestNodesCoords, nodelist=closestNodesCoords.keys(), node_size=10, node_color="black", alpha=0.4)
        nx.draw_networkx_nodes(graphToCluster, parentCoords, nodelist=parentCoords.keys(), node_size=20, node_color="black", alpha=0.25)

#%% Plotting
def PlotNewFigure(
        aspect=1,
        border=True,
        DPI=200,
        num=None,
        output=False,
        title=None,
        transparent=False,
        _hold=None
        ):
    """
    Start a new figure.
    
    Parameters
    ----------
    aspect : float (default = 1)
        Aspect ratio of the figure.
    border : bool (default = True)
        Control border around figure.
    DPI : int (default = 200)
        DPI of the figure.
    num : int, optional
        Set figure number.
    output : bool (default = False)
        Output fig and ax handles.
    title : string, optional
        Figure title.
    transparent : bool (default = False)
        Set figure transparency.
    """
    if _hold != None:
        aspect = _hold["aspect"]
        border = _hold["border"]
        DPI = _hold["DPI"]
        num = _hold["num"]
        title = _hold["title"]
        transparent = _hold["transparent"]
    
    # Start new figure
    # plt.figure(num, dpi=DPI) # old method
    fig, ax = plt.subplots(num=num, dpi=DPI) # set DPI
    plt.title(title)      # set title
    ax.set_aspect(aspect) # lock aspect ratio
    fig.tight_layout()    # set tight layout
    if border == False:   # remove border
        ax.axis('off')
    if transparent:       # enable transparency
        for item in [fig, ax]:
            item.patch.set_visible(False)
    if output:
        return fig, ax

def PlotGraph(
        graph, 
        arrowSize=10,
        axes=None,
        drawNodes=True,
        drawEdges=True,
        edgeAlpha=1,
        edgeAttribute="length",
        edgeColour='k',
        edgeCMap = None,
        edgeLabels=False,
        edgeLabelPos=0.5,
        edgeLabelSize=8,
        edgeStyle='-',
        edgeWidth=1,
        newFig=False,
        nodeAlpha=1,
        nodeColour="#1f78b4",
        nodeCMap=None,
        nodeLabels=False,
        nodeLabelsPos=None,
        nodeLabelSize=12,
        nodeSize=300,
        output=False
        ):
    """
    Plot a DiGraph object.
    
    Parameters
    ----------
    graph : FlexibleGraph, DiGraph
        Graph object.
    arrowSize : float (default = 10)
        Change the edges' arrow size.
    axes : axes type.
        Set axes to plot into.
    drawEdges : list (default = True)
        If specified, draw only the specified nodes.
    drawNodes : list (default = True)
        If specified, draw only the specified edges.
    edgeAlpha : float (0-1) (default = 1)
        Change alpha of edges.
    edgeAttribute : string (default = "length")
        Change edge attribute to display.
    edgeCMap : string, optional
        Colour map from MatPlotLib. For example "bwr" (blue, white, red). Requires edge attribute.
    edgeLabels : list (default = False)
        If specified, draw only the specified edge labels.
    edgeLabelPos : float (0-1) (default = 0.5 (mid))
        Change edge label positions along the edge.
    edgeLabelSize : float (default = 8)
        Change edge label size.
    edgeStyle : string (default = '-')
        Change edge style, for example '--' for dashed or ':' for dotted.
    edgeWidth : float (default = 1)
        Change the width of edges.
    newFig : bool (default = False)
        To create a new figure use newFig = True for default values (aspect=1, border=True, DPI=200, num=None, title=None, transparent=False).
    nodeAlpha : float (0-1) (default = 1)
        Change alpha of nodes.
    nodeColour : colour or array of colours (default='#1f78b4')
        Colour of nodes.
    nodeCMap : string, optional
        Colour map from MatPlotLib. For example "bwr" (blue, white, red)
    nodeLabels : list (default = False)
        If specified, draw only the specified node labels.
    nodeLabelsPos : dict (default = None)
        Modify position of node labels.
    nodeLabelSize : float (default = 12)
        Set size of node labels.
    nodeSize : int (default = 300)
        Change node size.
    output : bool (default = False)
        Output fig and ax handles.
    """
    coordList = GetCoordList(graph)
    
    # # Drawing everything (failed attempt)
    # nx.draw(graph, coordList, with_labels=nodeLabels, 
    #         font_size=nodeLabelSize,
    #         node_size=nodeSize,
    #         connectionstyle="arc3, rad = 0.0")
    #         # connectionstyle="arc3, rad = 0.15") # curve edges
    
    
    # Check hold parameter
    if newFig == True:
        fig, ax = PlotNewFigure(output=True)
    elif newFig != False:
        defaults = {"aspect":1, "border":True, "DPI":200, "num":None, "output":False, "title":None, "transparent":False}
        for figProp in ["aspect", "border", "DPI", "num", "title", "transparent", "output"]:
            try: newFig[figProp]
            except: newFig[figProp] = defaults[figProp]
        fig, ax = PlotNewFigure(_hold=newFig, output=True)

    # Draw all nodes
    if nodeCMap != None:
        nodeCMap = mpl.colormaps[nodeCMap]
    if drawNodes != False:
        if drawNodes ==True:
            drawNodes = coordList
        try:
            nx.draw_networkx_nodes(graph, pos=coordList, nodelist=drawNodes, node_size=nodeSize,
                                   alpha=nodeAlpha, node_color=nodeColour,
                                   cmap=nodeCMap, ax=axes)
        except: raise Exception("drawNodes should be a subset of the graph's nodes.")
    
    # Draw node labels if applicable
    if nodeLabels != False:
        if nodeLabelsPos == None:
            nodeLabelsPos = coordList
        
        if nodeLabels != True:
            labelsList = {k:k for k in nodeLabels}
        else:
            labelsList = {k:k for k in coordList}
            
        nx.draw_networkx_labels(graph, nodeLabelsPos, labels=labelsList,
                                font_size=nodeLabelSize, ax=axes)
    
    # Draw all edges
    if drawEdges != False:
        if drawEdges == True:
            edgeListTotal = GetEdgeList(graph)
        else: 
            edgeListTotal = drawEdges
        if edgeCMap != None:
            if type(edgeCMap) != str:
                colourCompensation = edgeCMap[1]
                edgeCMap = edgeCMap[0]
            else: colourCompensation = 1
            try: edgeCMap = mpl.colormaps[edgeCMap]
            except:
                try: edgeCMap = plt.cm[edgeCMap]
                except Exception as e: print(e)
                
            edgeColour = list(GetEdgeAttributeDict(graph, edgeAttribute).values())
            edgeColour = [x**colourCompensation for x in edgeColour]
                
        nx.draw_networkx_edges(graph, coordList, edgelist=edgeListTotal, arrowsize=arrowSize, alpha=edgeAlpha,
                                edge_color=edgeColour, edge_cmap=edgeCMap, width=edgeWidth, style=edgeStyle,
                                node_size=nodeSize, ax=axes, arrows=True) # remaining edges
    
    # Draw edge labels if applicable
    if edgeLabels != False and drawEdges != False:
        labels = nx.get_edge_attributes(graph, edgeAttribute)
        for idx in labels:
            label = GetEdgeList(graph)[GetEdgeList(graph).index(idx)]
            if label in edgeListTotal and\
               ((edgeLabels != True and label in edgeLabels) or edgeLabels == True):
                if type(labels[idx]) not in [list, set, dict, str]:
                    try:
                        labels[idx] = round(labels[idx], 1) # round of float values to 1 decimal
                    except Exception as e: print(e)
            else: 
                labels[idx] = ''
        nx.draw_networkx_edge_labels(graph, coordList, edge_labels=labels, 
                                      label_pos=edgeLabelPos,
                                      font_size=edgeLabelSize, ax=axes)
    
    # Output fig and ax handles
    if output:
        if newFig == False:
            fig, ax = plt.gcf(), plt.gca()
        return fig, ax
    
    test=GetGraphMap([52.3403, 4.8130, 52.2771, 4.7004])

G = ox.graph_from_place('Paris, France')
ox.plot_graph(G)