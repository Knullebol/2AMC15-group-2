import osmnx as ox
import pygame
import sys
from shapely.geometry import Polygon, LineString
import numpy as np


center = (51.447739, 5.488000)  # Latitude and longitude of TUe
dist = 250
WIDTH, HEIGHT = 1280, 860
BG_COLOR = (240, 240, 240)
ROAD_COLOR = (180, 180, 180)
STREET_COLOR = (120, 120, 120)
BUILDING_COLOR = (80, 80, 80)
FONT_COLOR = (250, 250, 250)


class Buildings:
    building_coors = []
    building_names = []
    
    
def draw_TUe_map(pygame, screen):

    # Download features
    G = ox.graph_from_point(center, dist=dist, network_type='drive')
    buildings = ox.features_from_point(center, dist=dist, tags={'building': True})
    cycleway = ox.features_from_point(center, dist=dist, tags={'cycleway:both': True})
    road = ox.features_from_point(center, dist=dist, tags={'highway': True})

    # Get bounds
    nodes = ox.graph_to_gdfs(G, edges=False)
    minx, miny, maxx, maxy = nodes.total_bounds
    if not buildings.empty:
        bminx, bminy, bmaxx, bmaxy = buildings.total_bounds
        minx, miny = min(minx, bminx), min(miny, bminy)
        maxx, maxy = max(maxx, bmaxx), max(maxy, bmaxy)
    if not cycleway.empty:
        pminx, pminy, pmaxx, pmaxy = cycleway.total_bounds
        minx, miny = min(minx, pminx), min(miny, pminy)
        maxx, maxy = max(maxx, pmaxx), max(maxy, pmaxy)

    def geo_to_screen(x, y):
        sx = int((x - minx) / (maxx - minx) * WIDTH)
        sy = int(HEIGHT - (y - miny) / (maxy - miny) * HEIGHT)
        return sx, sy

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Eindhoven Schematic Map (osmnx)")
    screen.fill(BG_COLOR)
    font = pygame.font.SysFont('calibri', 25, bold=True)


    def draw_text_with_border(surface, text, font, x, y, text_color):
        x = x - 50
        if text in ['nan']:
            return
        if len(text) >= 9:
            font = pygame.font.SysFont('calibri', 18, bold=True)
        else:
            font = pygame.font.SysFont('calibri', 25, bold=True)
        border_color = (0, 0, 0)
        text_surface = font.render(text, True, text_color)
        border_surface = font.render(text, True, border_color)
        # Offsets for border (left, right, up, down)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            surface.blit(border_surface, (x+dx, y+dy))
        # Draw main text
        surface.blit(text_surface, (x, y))
        
    # Draw buildings first
    if not buildings.empty:
        for idx, row in buildings.iterrows():
            geom = row.geometry
            if isinstance(geom, Polygon):
                pts = [geo_to_screen(x, y) for x, y in np.array(geom.exterior.coords)]
                pygame.draw.polygon(screen, BUILDING_COLOR, pts, 0)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    pts = [geo_to_screen(x, y) for x, y in np.array(poly.exterior.coords)]
                    pygame.draw.polygon(screen, BUILDING_COLOR, pts, 0)

    # Draw roads (edges)
    edges = ox.graph_to_gdfs(G, nodes=False)
    for _, row in edges.iterrows():
        color = STREET_COLOR if row.get('highway', '') in ['primary', 'secondary', 'tertiary', 'trunk'] else ROAD_COLOR
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            points = [geo_to_screen(x, y) for x, y in coords]
            pygame.draw.lines(screen, color, False, points, 6 if color == STREET_COLOR else 3)
        elif row.geometry.geom_type == 'MultiLineString':
            for linestr in row.geometry.geoms:
                coords = list(linestr.coords)
                points = [geo_to_screen(x, y) for x, y in coords]
                pygame.draw.lines(screen, color, False, points, 6 if color == STREET_COLOR else 3)

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
                    
    # Draw building text labels after all polygons
    if not buildings.empty:
        for idx, row in buildings.iterrows():
            geom = row.geometry
            name = row.get('name', f'B{idx}')
            if isinstance(geom, Polygon):
                cx, cy = geom.centroid.x, geom.centroid.y
                sx, sy = geo_to_screen(cx, cy)
                Buildings.building_coors.append((sx, sy))
                Buildings.building_names.append(name)
                #draw_text_with_border(screen, str(name), font, sx, sy, FONT_COLOR)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    cx, cy = poly.centroid.x, poly.centroid.y
                    sx, sy = geo_to_screen(cx, cy)
                    Buildings.building_coors.append((sx, sy))
                    Buildings.building_names.append(name)
                    #draw_text_with_border(screen, str(name), font, sx, sy, FONT_COLOR)

    return screen

