import xarray as xr
import pandas as pd
import datetime
from xml.dom import minidom
from urllib.request import urlopen, urlretrieve
from rasterio.plot import show
from distributed import Client, LocalCluster

xr.set_options(file_cache_maxsize=5)


cluster = LocalCluster(n_workers=5, threads_per_worker=1)
client = Client(cluster)

server_url = 'https://www.ncei.noaa.gov/thredds/'
request_url = 'cdr/ndvi/files/1982/'
opendap_url = 'dodsC/'

def get_elements(url, tag_name, attribute_name):
  """Get elements from an XML file"""
  usock = urlopen(url)
  xmldoc = minidom.parse(usock)
  usock.close()
  tags = xmldoc.getElementsByTagName(tag_name)
  attributes=[]
  for tag in tags:
    attribute = tag.getAttribute(attribute_name)
    attributes.append(attribute)
  return attributes


url = server_url + request_url + 'catalog.xml'
print(url)
catalog = get_elements(url, 'dataset', 'urlPath')
catalog = [x for x in catalog if x != '']


xr_ndvi_yr = xr.open_mfdataset([server_url + opendap_url + x for x in catalog],
                               chunks={'time': 1, 'longitude': -1, 'latitude': -1},
                               data_vars=['NDVI', 'QA'],
                               coords='all',
                               parallel=True,
                               compat='override',
                               join='override',
                               mask_and_scale=False)

test_list = []
for i in [server_url + opendap_url + x for x in catalog]:
    print(i)
    test = xr.open_dataset(i)
    test_list.append(test)

test2 = xr.merge(test_list)



# scale up
# try with distributed
#

min_x = -126.0
max_x = -100.0
min_y = 28.0
max_y = 49.0

xr_ndvi_yr = xr_ndvi_yr.sel(longitude=slice(min_x, max_x),
                            latitude=slice(max_y, min_y))


min_x = -126.0
max_x = -100.0
min_y = 28.0
max_y = 49.0

show(xr_ndvi_yr.sel(longitude=slice(min_x, max_x),
                    latitude=slice(max_y, min_y),
                    time=datetime.datetime(1982, 1, 2))['NDVI'].data)