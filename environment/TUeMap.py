from shapely.geometry import Polygon, LineString

import osmnx as ox
import numpy as np


CENTER = (51.447739, 5.488000)  # Latitude and longitude of TUe
RADIUS = 250                    # Radius in meters to fetch the map
WIDTH, HEIGHT = 1280, 860
BG_COLOR = (240, 240, 240)      # Background color of the map
ROAD_COLOR = (180, 180, 180)    # Color for roads
STREET_COLOR = (120, 120, 120)  # Color for main streets
BUILDING_COLOR = (80, 80, 80)   # Color for buildings
FONT_COLOR = (250, 250, 250)    # Color for text labels


class Buildings:
    building_coors = {}


def draw_TUe_map(pygame, screen):

    # Get buildings and roads
    G = ox.graph_from_point(CENTER, dist=RADIUS, network_type='drive')
    buildings = ox.features_from_point(CENTER, dist=RADIUS, tags={'building': True})
    road = ox.features_from_point(CENTER, dist=RADIUS, tags={'highway': True})

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
    pygame.display.set_caption("Eindhoven Schematic Map (osmnx)")
    screen.fill(BG_COLOR)

    # Draw buildings first
    if not buildings.empty:
        for idx, row in buildings.iterrows():
            geom = row.geometry
            if isinstance(geom, Polygon):
                pts = [geo_to_screen(x, y)
                       for x, y in np.array(geom.exterior.coords)]
                pygame.draw.polygon(screen, BUILDING_COLOR, pts, 0)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    pts = [geo_to_screen(x, y)
                           for x, y in np.array(poly.exterior.coords)]
                    pygame.draw.polygon(screen, BUILDING_COLOR, pts, 0)

    # Draw road
    if not road.empty:
        for idx, row in road.iterrows():
            if isinstance(row.geometry, LineString):
                coords = list(row.geometry.coords)
                points = [geo_to_screen(x, y) for x, y in coords]
                pygame.draw.lines(screen, ROAD_COLOR, False, points, 6)
                
            elif row.geometry.geom_type == 'MultiLineString':
                for linestr in row.geometry.geoms:
                    coords = list(linestr.coords)
                    points = [geo_to_screen(x, y) for x, y in coords]
                    pygame.draw.lines(screen, ROAD_COLOR, False, points, 6)

    # Store building coordinates and names into a dictionary
    if not buildings.empty:
        for idx, row in buildings.iterrows():
            geom = row.geometry
            name = row.get('name', f'B{idx}')
            if isinstance(geom, Polygon):
                cx, cy = geom.centroid.x, geom.centroid.y
                sx, sy = geo_to_screen(cx, cy)
                Buildings.building_coors[name] = (sx, sy)

            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    cx, cy = poly.centroid.x, poly.centroid.y
                    sx, sy = geo_to_screen(cx, cy)
                    Buildings.building_coors[name] = (sx, sy)

    return screen
