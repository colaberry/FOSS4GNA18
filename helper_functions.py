import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
from PIL import Image, ImageDraw
from skimage import measure 
from skimage.filters import threshold_otsu,laplace
import skimage
from skimage.morphology import closing, square,disk, extrema
import fiona
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import math
from descartes import PolygonPatch
from matplotlib.collections import LineCollection
import pylab as pl 
import matplotlib as mpl
from matplotlib.patches import Polygon
from scipy import ndimage
from skimage.measure import label
from skimage import exposure

# RGB(127,255,0)
def get_color(lower, upper, img): 
    
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask)
    return output

def get_bw(lower,upper, img ):
    mask = get_color(lower,  upper, img)
    gray = skimage.color.rgb2gray(mask)
    thresh = threshold_otsu(gray)
    bw = closing(gray > 0.0, disk(2))
   
    return bw

def get_alpha_shape(pts,alpha_value):     
    x  = [x[0] for x in pts] 
    y = [x[1] for x in pts]    
    pts_list = make_points(x,y)
    points_objects= [geometry.shape(point['geometry']) for point in pts_list]
    point_collection = geometry.MultiPoint(list(points_objects))
    concave_hull, edge_points = alpha_shape(points_objects, alpha=alpha_value)
    _ = plot_polygon(concave_hull)
    _ = plt.plot(x,y,'o', color='#f16824')
    return concave_hull,edge_points, x,y 


def show_points(bw):
    
    labeled = measure.label(bw)
    regions  = measure.regionprops(labeled)
    all_centroids= []
    for region in regions: 
        if region.area>1:
            all_centroids.append(region.centroid)
    plt.figure(figsize=(10,10))
    plt.imshow(bw)
    for i in range(len(all_centroids)):   
        plt.scatter(all_centroids[i][1],all_centroids[i][0], color='r' )
    plt.show()
    return all_centroids

def make_points(x,y):
    points_list = []
    for i in range(0,len(x)):
        points_list.append({'type': 'Feature', 'id': i, 'geometry': {'type': 'Point', 'coordinates': (x[i], y[i])}})
    return points_list

def plot_polygon(polygon):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .5

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig

def plot_multipolygon(polygon_list,img, tx = 0,ty= 0.0, plt_title= 'Multiploygon plot' ):
    
    fig = plt.figure(figsize= (10,10))
   
    ax = fig.add_subplot(111)
    plt.title(plt_title, fontsize=20)
    ax.set_ylim([0, 1000])
    
    ax.imshow(img, alpha =0.5)
    
    margin = 1.0
    for polygon in polygon_list:
        x_min, y_min, x_max, y_max = polygon.bounds

        ax.set_xlim([0, img.shape[1]])
        ax.set_ylim([ img.shape[0], 0])
        t2 = mpl.transforms.Affine2D().rotate_deg_around( img.shape[0]/2,  img.shape[1]/2, 90).translate(tx,ty) + ax.transData
        
        patch = PolygonPatch(polygon, fc='#999999',
                             ec='#000000', fill=True,
                             zorder=1, alpha= 0.5)
        patch.set_transform(t2)
        ax.add_patch(patch)
    plt.show()
   
    return None 


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

def point_intersect(pts,concave_hull ): 
    
    contains= []
    for pt in pts: 
        contains.append(concave_hull.contains(geometry.Point(pt[0],pt[1]))) 
        
    return contains

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def closest_polygon(points, polygon): 
    
    dist_list =[]
    for x in points: 

        point = geometry.Point((x[0], x[1]))
        pol_ext = geometry.LinearRing(polygon.exterior.coords)
        d = pol_ext.project(point)
        p = pol_ext.interpolate(d)
        closest_point_coords = list(p.coords)[0]
        dist_pt_poly= distance(np.array(closest_point_coords),np.array(list(point.coords)[0]))
        dist_list.append(dist_pt_poly)
    return dist_list

def add_points(maxima_points, polygon,threshold=5): 
    """
    Get closest points to a given polygon and keep those that are lower
    than a certain threshold. Return both the set of points and distances  
    """
    closest_points=  closest_polygon(maxima_points,polygon) 
    keep_index=np.where(np.asarray(closest_points)<threshold)
  
    if keep_index[0].shape[0]:
        return np.asarray(maxima_points)[keep_index], np.asarray(closest_points)[keep_index]
    else: 
        return None

def get_inside_points(x_interval=10, y_interval=10, polygon= None):
    u"""
    Perform sampling by substituting the polygon with a regular grid of
    sample points within it. The distance between the sample points is
    given by x_interval and y_interval.
    """

    samples =[]
    ll = polygon.bounds[:2]
    ur = polygon.bounds[2:]
    low_x = int(ll[0]) / x_interval * x_interval
    upp_x = int(ur[0]) / x_interval * x_interval + x_interval
    low_y = int(ll[1]) / y_interval * y_interval
    upp_y = int(ur[1]) / y_interval * y_interval + y_interval

    for x in floatrange(low_x, upp_x, x_interval):
        for y in floatrange(low_y, upp_y, y_interval):
            p = geometry.Point(x, y)
            if p.within(polygon):
                samples.append(p)
    
    inside_pts = [list(x.coords) for x in samples]
    inside_pts_coords = [x[0] for x in inside_pts]

    return inside_pts_coords

def floatrange(start, stop, step):
    while start < stop:
        yield start
        start += step
  

def  get_maxima_points(boundary_points, img, sigma=0.3 ): 
    width = img.shape[0]
    height = img.shape[1]

    canvas = Image.new('L', (width, height), 0)
    ImageDraw.Draw(canvas).polygon(boundary_points, outline=1, fill=1)
    mask = np.array(canvas)
    
    transformed_image = ndimage.distance_transform_edt(mask)
    normalized_ti =transformed_image/np.max(transformed_image)
    laplacian2 = ndimage.gaussian_laplace(normalized_ti, sigma=sigma)
   
    plt.figure(figsize=(10,10))
    plt.imshow(laplacian2)
    plt.colorbar()
    plt.show()

    local_maxima = extrema.local_maxima(1-laplacian2)
    label_maxima = label(local_maxima)
    plt.figure(figsize=(10,10))
    plt.imshow(label_maxima)
    plt.colorbar()
    plt.show()
    np.unique(label_maxima)
    all_inside_y, all_inside_x = np.where(label_maxima>1)
    all_points =[]
    for x,y in zip(all_inside_x, all_inside_y): 
          all_points.append(tuple((x,y)))
    return all_points 


def non_intersect(points_list,*polygonal_shape):
    bool_list = []
    final_bool = points_list.copy()
    final_points = []
    for i in range(0,len(polygonal_shape)):
        bool_list.append(point_intersect(points_list,polygonal_shape[i]))
    for i in range(0,len(points_list)):
        for j in range(0,len(polygonal_shape)):
            if bool_list[j][i] == True:
                final_bool[i] = True
                break
            else:
                final_bool[i] = False
    for i in range(0,len(points_list)):
        if final_bool[i] == False:
            final_points.append(points_list[i])
    return final_points




def all_points(*points_list):
    all_pts = []
    for i in points_list:
        for a in i:
            all_pts.append(a)
    return all_pts

def get_unmapped_area(full_convex_hull,*polygons):
    shapes = []
    unmapped_area = full_convex_hull
    for i in polygons:
        shapes.append(full_convex_hull.symmetric_difference(i))
    for a in shapes:
        unmapped_area = unmapped_area.intersection(a)
    return unmapped_area

    