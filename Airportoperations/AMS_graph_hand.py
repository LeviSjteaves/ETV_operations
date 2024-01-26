# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:20:54 2023

@author: levi1
"""

import networkx as nx
from math import sqrt
from pyproj import CRS, Transformer
import pandas as pd
import matplotlib.pyplot as plt

def calculate_distance(coord1, coord2):
    return sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)

def translate_coordinates_to_meters(node_coordinates):
    in_crs = CRS('epsg:4326')  # WGS 84
    out_crs = CRS('epsg:28992')  # Web Mercator

    transformer = Transformer.from_crs(in_crs, out_crs)
    
    for node, (lat, lon) in node_coordinates.items():
        x, y = transformer.transform(lat, lon)
        node_coordinates[node] = (x, y)
         
        

def create_airport_graph(node_coordinates, edges):
    translate_coordinates_to_meters(node_coordinates)

    node_coordinates_df = pd.DataFrame.from_dict(node_coordinates, orient='index').reset_index()
    node_coordinates_df.columns = ['node','x', 'y']
    del node_coordinates_df['node']
    
    G = nx.MultiDiGraph()

    # Add nodes with coordinates
    for node, (x, y) in node_coordinates.items():
        G.add_node(node, pos=(x, y))

    # Add directed edges with lengths based on coordinates
    for edge in edges:
        u, v, bidirectional = edge
        if u in node_coordinates and v in node_coordinates:
            distance = calculate_distance(node_coordinates[u], node_coordinates[v])
            G.add_edge(u, v, weight=distance)
            if bidirectional == True:
                G.add_edge(v, u, weight=distance)
    return G

# Define the coordinates of nodes
node_coordinates = {
    #0: (52.36249520205651, 4.711999569734029),
    #1: (52.343904120549176, 4.710238968255857),
    #2: (52.338550003247036, 4.709594969144059),
    3: (52.3341215252892, 4.709371192254495),
    #4: (52.32884138143807, 4.70884807896224),
    #5: (52.33601400693928, 4.712331382943056),
    6: (52.3325890537682, 4.712030578243304),
    7: (52.331107340452505, 4.714498210389625),
    8: (52.32965139252123, 4.717334301610054),
    9: (52.33043074890147, 4.718478134840405),
    10: (52.32835564189358, 4.722287763278481),
    11: (52.32164439752322, 4.732464196425797),
    12: (52.32092952528239, 4.7335017071545264), 
    13: (52.33086072641612, 4.744356836941741),
    14: (52.30072972374922, 4.732971586874934),
    15: (52.30200803764744, 4.734707164932471),
    16: (52.3020038565473, 4.7373359906253905),
    17: (52.304222404301605, 4.7375391211385045),
    18: (52.30782832257149, 4.737868229801696),
    #19: (52.31296972951351, 4.738343905504889), 
    #20: (52.31757908557356, 4.738755095586184),
    #21: (52.3225542007504, 4.739210029607778),
    22: (52.330709949999914, 4.739973219951756),
    23: (52.33093761786676, 4.742904537719856),
    24: (52.3266400791422, 4.74231042862517),
    25: (52.314835445212914, 4.741439412203655),
    26: (52.3138868517011, 4.741343489464057),
    27: (52.31127021320595, 4.741083589274372),
    28: (52.30911545806042, 4.740933143501695),
    29: (52.30596298773835, 4.740654396979596), 
    30: (52.30407608464397, 4.740484528440805),
    31: (52.32035393172792, 4.741938194116353),
    32: (52.326721697350116, 4.744016536040103),
    33: (52.32028197181789, 4.743362969197418),
    34: (52.3147416923667, 4.742857832419291),
    35: (52.31391173553386, 4.742799648418822), 
    36: (52.309075839855495, 4.742334770197672),
    37: (52.30385653441757, 4.741819277898517),
    38: (52.3207127446842, 4.746353941038429),    
    39: (52.31408755493464, 4.746255841646516),
    40: (52.31424377792806, 4.754340371932932),
    41: (52.31445114124245, 4.759239446007616),
    42: (52.31462156489886, 4.764176120392282),
    43: (52.31474228125844, 4.766708343959003),
    44: (52.31483459354655, 4.771935410954531),
    45: (52.31580741134088, 4.7753039658101075),
    46: (52.31575060505035, 4.772098031386184),
    47: (52.31554468163681, 4.766685112780461),
    48: (52.315530479986705, 4.765302843952755), 
    49: (52.31528195037395, 4.759169751688046),
    50: (52.31512573103878, 4.75433761835191),
    51: (52.314870098224716, 4.746682868652959),
    52: (52.3111419518015, 4.7455677609051445),
    53: (52.31158934594357, 4.7546860893295415),
    54: (52.31278947617378, 4.759402065736491), 
    55: (52.31284628626489, 4.764594285618714),
    56: (52.31090049911015, 4.768764324053212),
    57: (52.31281788124728, 4.775106498700818),    
    58: (52.31281093251305, 4.776367561673974),
    59: (52.3113946699831, 4.77633422962598),
    60: (52.31074256264751, 4.7748009554182484),
    61: (52.308348023624994, 4.774634295178278),
    62: (52.308307264312546, 4.776034241205975), 
    63: (52.306524011171206, 4.774417636878259),
    64: (52.305780118629656, 4.7753176021741),
    65: (52.30587183205584, 4.77296769259044), 
    66: (52.30528078645997, 4.773501005358346),
    67: (52.30456744506054, 4.771751072838653),
    68: (52.305229833884, 4.771151095974759),
    69: (52.30375218362704, 4.766951257717472),
    70: (52.301917790621296, 4.762518095334251), 
    71: (52.300572520783795, 4.759251554630825),
    72: (52.29821820016163, 4.752568478700426),
    73: (52.297606667850765, 4.753101791468333), 
    74: (52.299950829185065, 4.759668204923179),
    75: (52.30127573505169, 4.763001409722593),
    76: (52.302987862424615, 4.767567900626268),
    77: (52.29971507292495, 4.749438827086006),
    78: (52.302705487963586, 4.7413310651714955),
    79: (52.29560886326246, 4.746029553608001),
    80: (52.30105064738469, 4.755339871010836),
    81: (52.3036476256882, 4.760722432797437),
    82: (52.305230053431025, 4.764075605025912),
    83: (52.306795410104, 4.767957492847625),
    84: (52.30850061188689, 4.771339256201406),
    85: (52.316659933223114, 4.7466429165417825),
    #86: (52.31721846389905, 4.762702124502585),
    #87: (52.31743307066417, 4.768718957953477),
    88: (52.31769729026173, 4.776955080987146),
    89: (52.3183852164465, 4.796687764772861),
    #90: (52.310966065332366, 4.768923736421211),
    91: (52.313882358363614, 4.787156854123504),
    92: (52.32112769251477, 4.780120552198136),
    93: (52.316049047584514, 4.783031528384413),
    94: (52.31361797819573, 4.782289246326253),
    #95: (52.30905473930981, 4.778981179615933),
    #96: (52.29099085621545, 4.777361452631464),
    97: (52.31426792659085, 4.795516296528264),
    98: (52.311941588487926, 4.7999144622187275),
    99: (52.31034159462053, 4.802918868722039),
    100: (52.30530688055778, 4.790538806219759),
    101: (52.30124875444256, 4.784756470217245),
    102: (52.30184154229707, 4.790108526006406),
    103: (52.30228712230842, 4.775420957474423),
    104: (52.292200103544815, 4.745043054769567),
    105: (52.29589416945909, 4.763983129520848),
    106: (52.29412728967028, 4.755848910753563),
    107: (52.29213959519698, 4.752199051498203),
    108: (52.30483836782026, 4.7778764032013985),
    155: (52.33467399894124, 4.7364683256413835),
    156: (52.3342652888607, 4.744610016781785),
    
}
node_coordinates_e = {
    # Only ETV nodes
    3: (52.3341215252892, 4.709371192254495),
    17: (52.304222404301605, 4.7375391211385045),
    22: (52.330709949999914, 4.739973219951756),
    52: (52.3111419518015, 4.7455677609051445),
    53: (52.31158934594357, 4.7546860893295415),
    54: (52.31278947617378, 4.759402065736491), 
    55: (52.31284628626489, 4.764594285618714),
    56: (52.31090049911015, 4.768764324053212),
    79: (52.29560886326246, 4.746029553608001),
    80: (52.30105064738469, 4.755339871010836),
    81: (52.3036476256882, 4.760722432797437),
    82: (52.305230053431025, 4.764075605025912),
    83: (52.306795410104, 4.767957492847625),
    84: (52.30850061188689, 4.771339256201406),
    89: (52.3183852164465, 4.796687764772861),
    98: (52.311941588487926, 4.7999144622187275),
    92: (52.32112769251477, 4.780120552198136),
    101: (52.30124875444256, 4.784756470217245),
    104: (52.292200103544815, 4.745043054769567),
    105: (52.29589416945909, 4.763983129520848),
    107: (52.29213959519698, 4.752199051498203),
    108: (52.30483836782026, 4.7778764032013985),
    
    109: (52.33399287313866, 4.713211632287101),
    110: (52.32222460535619, 4.734000463731324),
    111: (52.32211973693743, 4.736540784554844),
    112: (52.335397311889636, 4.738196263064779),
    113: (52.33525692670541, 4.74261022309684),
    114: (52.30094112005267, 4.7520101607536125),
    115: (52.30378777611243, 4.759876904488094),
    116: (52.30545043196897, 4.761589008791205),
    117: (52.306796387471366, 4.765137224277812),
    118: (52.308660311457956, 4.765606592937114),
    119: (52.310254400582814, 4.765358335553927),
    120: (52.31169629503089, 4.762268433029724),
    121: (52.31155113882058, 4.757514329133435),
    122: (52.30924700965131, 4.752427506036307),
    123: (52.30871035169881, 4.744126068390447),
    124: (52.31801517008099, 4.744550593027053),
    125: (52.32334333207428, 4.7445541951116885),
    126: (52.33388732620329, 4.745492634496264),
    127: (52.33117251577059, 4.742237579066265),
    128: (52.317900783670886, 4.7409822352066575),
    129: (52.317662421253594, 4.735758072229973),
    130: (52.314370842753995, 4.743732928023947),
    131: (52.31484298683346, 4.757443260969438),
    132: (52.30442301627138, 4.734977483537876),
    133: (52.315314063421475, 4.772663378899375),
    134: (52.319729419899446, 4.77534391967623),
    135: (52.32102574090801, 4.782367710403276),
    136: (52.31869702885119, 4.800891675209116),
    137: (52.3131995614938, 4.80501229950848),
    138: (52.31291275689646, 4.79752417224428),
    139: (52.30015910642428, 4.78020223918299),
    #140: (52.29614368887837, 4.78184503964404),
    141: (52.30015284425981, 4.7902739676865425),
    142: (52.308028313018866, 4.799532633444122),
    143: (52.30025737440668, 4.773013379260476),
    144: (52.297478606755746, 4.769727383862434),
    145: (52.30323541015229, 4.769903733264814),
    146: (52.2933371475513, 4.758499545578859),
    147: (52.29879404889008, 4.752956690831788),
    148: (52.31718649597989, 4.803692170780639),
    149: (52.31366710558121, 4.775606487637299),
    150: (52.30611681203356, 4.774618541818956),
    151: (52.302187853517395, 4.764374163817013),
    152: (52.298303438153965, 4.754416237523767),
    #153: (52.362667444739486, 4.711953031764774),
    #154: (52.290832659134914, 4.777345045207053),
}

# Define the edges
edges = [(1, 5, True),
         (2, 5, True), (5, 6, True), (3, 6, True), (4, 7, True), (6, 7, True),
         (7, 8, False), (9, 7, False), (8, 10, False), (10, 9, False), (10, 11, True),
         (11, 12, True), (11, 155, True), (12, 14, True), (14, 15, True), (15, 16, True),
         (15, 17, True), (22, 23, True), (155, 156, True), (156, 13, True),
         (23, 24, False), (24, 31, False), (31, 25, False), (25, 26, False),
         (26, 27, False), (27, 28, False), (28, 29, False), (29, 30, False),
         (21, 24, True), (20, 25, True), (19, 27, True), (19, 28, True), (18, 29, True),
         (17, 30, True), (12, 31, True), (13, 23, True),
         (32, 13, False), (33, 32, False), (34, 33, False), (35, 34, False),
         (36, 35, False), (37, 36, False), (24, 32, True), (31, 33, True),
         (34, 25, True), (26, 35, True), (28, 36, True), (30, 37, True), (14, 78, True), (78, 37, True),
         (33, 38, True), (35, 39, False), (39, 40, False), (40, 41, False), (41, 42, False),
         (42, 43, False), (43, 44, False), (45, 46, False), (46, 47, False), (47, 48, False), (48, 49, False),
         (49, 50, False), (50, 51, False),  (51, 34, False), (56, 44, True), (42, 55, True), (43, 55, True), (41, 54, True), (40, 53, True),
         (40, 53, True), (39, 52, True),(56, 60, True), 
         (51, 39, True), (50, 40, True), (49, 41, True), (48, 42, True), (47, 43, True), (46, 44, True),
         (44, 57, False), (57, 60, False), (60, 61, False), (62, 63, False),
         (63, 65, False), (65, 68, False), (68, 69, False), (69, 70, False), (70, 71, False), (71, 72, False),
         (73, 74, False), (74, 75, False), (75, 76, False), (76, 67, False),
         (67, 66, False), (66, 64, False), (64, 62, False), (62, 59, False), (62, 59, False), (61, 63, False),
         (59, 58, False), (58, 45, False),  (72, 73, True), (71, 74, True), (70, 75, True), (69, 76, True), (68, 67, True), (65, 66, True),
         (63, 64, True), (61, 62, True), (60, 59, True), (57, 58, True), (77, 37, False), (78, 77, False), (72, 77, True), 
         (16, 30, True), (30, 78, True), (72, 79, False), (79, 73, False),  (80, 72, True), (80, 71, True),
         (70, 81, True), (69, 82, True), (68, 83, True), (63, 83, True), (65, 83, True), (61, 84, True), 
         (51, 85, True), (86, 49, True), (87, 47, True), (87, 46, True), (45, 88, True), (88, 92, True), 
         (45, 93, True), (93, 91, True), (91, 97, True), (97, 89, True), (58, 94, True), (95, 59, True), (94, 91, True), 
         (79, 104, True), (73, 106, True), (106, 107, True), (106, 105, True), (105, 103, True), (103, 64, True), (103, 67, True),
         (108, 63, True), (108, 64, True), (108, 101, True), (101, 102, True), (102, 100, True), (100, 94, True), (100, 99, True),
         (99, 98, True), (98, 97, True), 
         # Add more edges as needed
]

edges_e = [
    (3, 109, True), (109, 110, True), (110, 111, True), (111, 112, True), (112, 113, True),
    (113, 126, True), (126, 125, True), (125, 124, True), (124, 130, True), (123, 130, True), (123, 122, True),
    (122, 121, True), (121, 120, True), (120, 119, True), (119, 118, True), (118, 117, True), (117, 116, True),
    (116, 115, True), (115, 114, True), (114, 123, True), (132, 129, True), (129, 128, True), (128, 124, True), 
    (129, 111, True), (113, 127, True), (127, 128, True), (127, 22, True), (17, 132, True), (123, 52, True), (122, 53, True), 
    (54, 121, True), (55, 120, True), (56, 119, True), (118, 84, True), (117, 83, True), (116, 82, True), (115, 81, True), 
    (114, 80, True), (130, 131, True), (131, 133, True), (124, 134, True), (134, 92, True), (135, 92, True), (135, 136, True), (136, 148, True), 
    (148, 138, True), (138, 98, True), (98, 137, True), (137, 142, True), (142, 141, True), (141, 140, True), (140, 139, True), 
    (141, 139, True), (140, 139, True), (143, 144, True), (89, 136, True), (138, 98, True), (98, 137, True), 
    (143, 139, True), (139, 101, True), (146, 147, True), (105, 144, True), (105, 146, True),(147, 79, True), (107, 146, True), 
    (143, 108, True), (108, 150, True), (82, 151, True), (133, 56, True), (131, 54, True), (138, 139, True),
    (133, 149, True), (149, 150, True), (150, 151, True), (151, 152, True), (152, 146, True), (152,114 , True), (114, 147, True), 
    (143,145 , True), (145, 151, True),  (148, 137, True)
    ]

# Create and visualize the graph
airport_graph = create_airport_graph(node_coordinates, edges)
airport_graph_e = create_airport_graph(node_coordinates_e, edges_e)

node_positions = nx.get_node_attributes(airport_graph, 'pos')
node_positions_e = nx.get_node_attributes(airport_graph_e, 'pos')

fig, ax = plt.subplots()
nx.draw(airport_graph, pos=node_positions, with_labels=True, node_size= 25, font_size= 12)
nx.draw(airport_graph_e, pos=node_positions_e, with_labels=True, node_color='red',  node_size= 25, font_size= 12, edge_color='grey')

# Save the MultiGraph to a file
file_path = "EHAM_graph_hand_a.gpickle"
nx.write_gpickle(airport_graph, file_path)
# Save the MultiGraph to a file
file_path = "EHAM_graph_hand_e.gpickle"
nx.write_gpickle(airport_graph_e, file_path)
