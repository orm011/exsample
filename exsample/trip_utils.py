from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import object
from past.utils import old_div
from IPython.core.display import HTML

from . import avutils
import os.path
import pandas as pd
import shapely
import glob
from collections import OrderedDict
import string


def to_secs(x):
    bname = os.path.basename(x)
    pts = bname.split('.')
    for pt in pts:
        if set(list(pt)).issubset(set(string.digits)):
            return pt
    assert False

def parse_timestamp_from_filename(x, clipped=True):
    secs = float(to_secs(x))
    if clipped:
        secs=int(secs)
    return pd.to_datetime(secs, unit='s')

def get_epoch(file_path):
    return file_path.map(to_secs).map(lambda x: float(x)*1000*1000*1000).map(pd.to_datetime)

def read_videos_txt(dirpath):
    vds = pd.read_csv(dirpath + '/videos.txt',
                      parse_dates=[[0, 1]],
                      delim_whitespace=True,
                      header=None)\
            .rename(index=str, columns={'0_1': 'date',
                                        2: 'byte_size',
                                        3: 's3path'})
    vds['timestamp'] = get_epoch(vds.s3path)
    vds.sort_values('timestamp').reset_index(drop=True)
    return vds


def github_plot_geojson(geojsonstr, height=int(420*1.3), width=int(620*1.3)):
    import github

    g = github.Github(open('/home/orm/.ghb').read())
    u = g.get_user()
    K_DESC='geojsonplot'
    K_NAME='geojson'
    c = github.InputFileContent(content=geojsonstr)
    gi = u.create_gist(public=True, description=K_DESC, files={K_NAME:c})
    url = gi.raw_data['files'][K_NAME]['raw_url']
    API_URL = 'https://render.githubusercontent.com/view/geojson'
    html_string= ("<iframe height='{height}' width='{width}' frameborder='0' src='{api_url}?url={url}'></iframe>"
        .format(api_url=API_URL, url=url, height=height, width=width))
    return HTML(html_string)


# example filenames:
# F97D34AC-3D84-447F-A22C-3D495E7578A8.1517172275.51304.mov
# F97D34AC-3D84-447F-A22C-3D495E7578A8.1530372592.MP4
def get_clip_info(dirpath, pattern=None, verbose=False):
    if pattern is None:
        movs = glob.glob(dirpath+'/*mov')
        MP4s = glob.glob(dirpath+'/*MP4')
        mp4s = glob.glob(dirpath+'/*mp4')
        vids = movs + mp4s + MP4s
    else:
        vids = glob.glob(dirpath+'/' + pattern)

    globs = pd.DataFrame(data=vids, columns=['path'])
    if globs.empty:
        if verbose:
            print('Warning: no mov files found here')

    globs['timestamp'] = get_epoch(globs.path)
    globs['ts_string'] = globs.path.map(to_secs)

    globs = globs.sort_values('timestamp').reset_index(drop=True)

    globs['vmeta'] = globs.path.map(lambda x: avutils.get_video_meta(x))
    globs['valid'] = globs['vmeta'].map(lambda x : x.valid if x is not None else None)
    globs['frame_count'] = globs['vmeta'].map(lambda  x: x.frame_count if x is not None else None)
    globs['duration'] = globs['vmeta'].map(lambda x : pd.to_timedelta(x.duration, unit='s') if x is not None else None)


    if not globs.valid.all():
        if verbose:
            print ('warning: some of the videos cannot be opened')
            print(globs.path[~ globs.valid])


    time_diff = globs.timestamp.diff().shift(-1)
    discrepancy =  old_div(time_diff.map(lambda x: x.total_seconds()).iloc[:-1], \
                   globs.duration.map(lambda x: x.total_seconds()).iloc[:-1])
    tmp = globs
    #tmp['discrepancy'] = discrepancy

    tmp['time_diff'] = time_diff
    tmp['time_gap'] = (tmp.time_diff - tmp.duration).map(lambda x: x.total_seconds())
    # by_disc = tmp.sort_values('discrepancy', ascending=False)
    # print by_disc

    if verbose:
        if not globs.empty:
            if (discrepancy.dropna() > 1.1).any():
                print ('warning: seem to be missing a video? ')
                #by_disc[['path', 'duration', 'time_diff']].head()
            if (tmp.time_gap < 0).any():
                print ('warning: there is a negative gap in one of the videos')

    return tmp


import shapely.geometry as geom

def to_point_series(df, lon_lat_columns=[u'lon',u'lat']):
    return df[lon_lat_columns].apply(tuple,1).map(lambda x : geom.Point(x[0], x[1])  if x[0] is not None else None)

def read_telemetry(path, columns=[]):
    t= pd.read_csv(path)
    t['timestamp'] = pd.to_datetime(t.time * 1000000)
    t = t.rename(axis=1, mapper=lambda m: m.replace('(m/s)', '_mps'))
    t = t.set_index('timestamp')

    if columns == []:
        return t
    else:
        return t[columns]


class Telemetry(object):
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path)
        self.data = self.data.rename(axis=1, mapper=lambda m: m.replace('(m/s)', '_mps'))
        # self._init_telemetry_extra_columns()
        self.start = self.data.timestamp.min()
        self.end = self.data.timestamp.max()

    def _init_telemetry_extra_columns(self):
        import geopandas as gpd
        import geojson
        t = self.data
        t['timestamp'] = pd.to_datetime(t.time * 1000000)
        t['location'] = to_point_series(t, lon_lat_columns=['lon', 'lat'])
        t['location'] = self.data[['gps_valid', 'location']].apply(lambda x: None if x[0] == 0 else x[1], 1)
        t['mm_loc'] = to_point_series(t, lon_lat_columns=['mm_lon', 'mm_lat'])
        self.data = gpd.GeoDataFrame(self.data, geometry=t['mm_loc'])

    def get_trajectory_geojson(self):
        import geojson
        dt = self.data
        name = os.path.basename(self.path)
        coords = dt[dt.gps_valid == 1][['lon', 'lat']].apply(tuple, axis=1)
        ls = shapely.geometry.LineString(coordinates=coords)
        trip_line = geojson.Feature(geometry=shapely.geometry.mapping(ls), properties=dict(file_name=name))
        start_time = dt.timestamp.iloc[0]
        duration = dt.timestamp.iloc[-1] - start_time
        trip_properties = OrderedDict([('file_name', name),
                                       ('start_time', str(start_time.round('60s'))),
                                       ('duration', str(duration.round('60s'))),
                                       ('pct_gps_valid', '%0.1f' % (dt.gps_valid.mean() * 100)),
                                       ('pct_accel_valid', '%0.1f' % (dt.accel_valid.mean() * 100)),
                                       ('mm_dist_km', dt.mm_dist_km.max()),
                                       ])
        start_pt = geojson.Feature(geometry=geojson.Point(coordinates=coords.iloc[0]),
                                   id='start',
                                   properties=trip_properties)
        fc = geojson.FeatureCollection(features=[trip_line, start_pt])
        line_gjs = geojson.dumps(fc)
        return line_gjs

    def plot_trajectory(self):
        line_gjs = self.get_trajectory_geojson()
        return github_plot_geojson(line_gjs)

from . import box_utils


class CarTrip(object):
    def __init__(self, ios_trip):
        self.ios_trip = ios_trip
        self.folder = os.path.dirname(ios_trip.telemetry_path)
        self.clip_metadata = get_clip_info(self.folder, 'wnum*MP4')
        assert self.clip_metadata.timestamp.is_monotonic_increasing
        self.video_start = self.clip_metadata.timestamp.min()
        # self.video_end = self.clip_metadata.timestamp.iloc[-1] + self.clip_metadata.duration.iloc[-1]
        self.telemetry = Telemetry(ios_trip.telemetry_path)
        self.vf = avutils.VideoFile(path=ios_trip.video_path)

    def get_box_overlay(self, class_name, **kwargs):
        boxes = box_utils.get_boxes_df(self.ios_trip, class_names=[class_name], **kwargs)
        return box_utils.BoxOverlay(boxes_df=boxes, video_file=self.vf)

    def compare_timelines(self):
        ## should be: quick viz to check timestamp alignment (clip by clip, and telemetry)
        ## we have video timestamps and telemetry timestamps we can compare
        pass