from shapely.geometry import Polygon
import pygame
import osmnx as ox
import numpy as np


CENTER = (51.447892, 5.486253)  # Latitude and longitude of TUe
RADIUS = 180                    # Radius in meters to fetch the map
WIDTH, HEIGHT = 400, 300
BG_COLOR = (240, 240, 240)      # Background color of the map
BUILDING_COLOR = (80, 80, 80)   # Color for buildings
ROAD_COLOR = (160, 160, 160)    # Color for roads
ROAD_SIZE = 8
ROAD_EDGES = [                  # Extra predefined road to draw
    [(164, 82), (164, 217)],
    [(276, 71), (276, 83)],
    [(90, 84), (90, 214)],
    [(308, 174), (163, 174)],
    [(166, 60), (184, 60)],
    [(315, 26), (315, 82)],
    [(309, 83), (400, 83)],
    [(65, 124), (113, 124)],
]


class Buildings:
    """
    Helper class to store building coordinates and names for matplotlib label
    """
    building_coors = {}


def draw_TUe_map(pygame: pygame, screen: pygame.Surface):
    """
    Draws a schematic map of the TUe area.
    OSMnx fetches map elements, and pygame renders all of them.
    """
    # Get buildings and roads
    G = ox.graph_from_point(CENTER, dist=RADIUS, network_type='drive')
    buildings = ox.features_from_point(CENTER, dist=RADIUS, tags={'building': True})
    road = ox.features_from_point(CENTER, dist=RADIUS, tags={'highway': ['unclassified', 'cycleway']})

    # Get bounds of the graph, and intialize the bounding box
    nodes = ox.graph_to_gdfs(G, edges=False)  # convert nodes to GeoDataFrame
    minx, miny, maxx, maxy = nodes.total_bounds

    if not buildings.empty:
        bminx, bminy, bmaxx, bmaxy = buildings.total_bounds

        # adjust the bounding box to just include visible buildings
        minx, miny = min(minx, bminx), min(miny, bminy)
        maxx, maxy = max(maxx, bmaxx), max(maxy, bmaxy)

    def geo_to_screen(x, y):
        """Convert geographic coordinates to screen coordinates."""
        sx = int((x - minx) / (maxx - minx) * WIDTH)
        sy = int(HEIGHT - (y - miny) / (maxy - miny) * HEIGHT)
        return sx, sy

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TU/e Map")
    screen.fill(BG_COLOR)

    # Draw buildings first
    if not buildings.empty:
        for idx, row in buildings.iterrows():
            geom = row.geometry
            name = row.get('name', f'B{idx}')
            if isinstance(geom, Polygon) and not isinstance(name, float):
                pts = [geo_to_screen(x, y)
                       for x, y in np.array(geom.exterior.coords)]
                pygame.draw.polygon(screen, BUILDING_COLOR, pts, 0)

    # Draw road
    for edge in ROAD_EDGES:
        pygame.draw.lines(screen, ROAD_COLOR, False, edge, ROAD_SIZE if edge != [(308, 174), (163, 174)] else 6)
    if not road.empty:
        for idx, row in road.iterrows():
            if isinstance(row['name'], float):
                continue
            geom = row.geometry
            pygame.draw.lines(screen, ROAD_COLOR, False,
                              [geo_to_screen(x, y) for x, y in np.array(geom.coords)], ROAD_SIZE)
                
    # Store building coordinates and names into a dictionary
    if not buildings.empty:
        for idx, row in buildings.iterrows():
            geom = row.geometry
            name = row.get('name', f'B{idx}')
            if isinstance(geom, Polygon):
                cx, cy = geom.centroid.x, geom.centroid.y
                sx, sy = geo_to_screen(cx, cy)
                Buildings.building_coors[name] = (sx, sy)

    return screen
