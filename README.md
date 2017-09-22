# UFO_Sightings
Scraped data from www.nuforc.org

This is a series of .py files for scraping, visualizing, and analyzing U.F.O. data from www.nuforc.org. .csv files are also provided. I apologize in advance as code readibility and efficiency are not optimized. Files can be run as is in python 2.7.

## DATA FILES

### uforeports.csv
To obtain the scraped results .csv file run the following commands from your terminal:
1. `cd /path/to/files/directory`
2. `zip -F splituforeports.zip --out uforeports.zip`
3. `unzip uforeports.zip`

You should have a file named uforeports.csv which contains raw U.F.O. sightings entries scraped from nuforc.org. You're welcome to scrape your own files using `UFO_Spider.py`.

### Coordinates_Full.csv
Contains latitude and longitude coordinates for sightings. These results were obtained using geopy.

### Airport_Data.csv
Latitude and longitude coordinates for airports, heliports, and seaplane bases across the United States. Spreadsheet sliced from data obtained from <https://www.faa.gov/airports/airport_safety/airportdata_5010/>.

### Military_Data.csv
Latitude and longitude coordinates for military bases across the United States. Obtained from <https://www.google.com/maps/d/viewer?mid=1hvB-oq9gE0H8gEwKJ4XHFOKaY5k&hl=en_US&ll=49.67042577190774%2C-117.29381349999994&z=2>

Download the KML file and convert it to CSV using software of your choice.

### Other CSV files
Other CSV files will be created when running code from the python scripts. You are more than welcome to use your own code to do whatever you'd like with the scraped data file.

## Python Code

### 1. UFO_Spider.py

Code used for scraping U.F.O. sightings data from www.nuforc.org. You may scrape by shape by altering the script yourself.

### 2. UFO_Processing.py

Ugly code I wrote to clean up and organize the scraped text entries.

### 3. UFO_Sightings.py

Code for shape, duration, and datetime data visualization.

### 4. UFO_Maps.py

Code for location data visualization. Code for Coordinates_full.csv can be found in `UFO_Processing.py`. This bit takes a long time to run, so I commented it out within the code.

```
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from time import sleep

geolocator = Nominatim()
places = data['place'] 
lat, lon = [],[]
for k in range(0,len(places)):
    location = geolocator.geocode("%s" % places[k], timeout = 10) 
    sleep(1.25) # Be courteous with this parameter or they'll kick you!
    try:
        lat.append(location.latitude)
        lon.append(location.longitude)
    except (AttributeError, GeocoderTimedOut):
        lat.append(np.nan)
        lon.append(np.nan)

coords = pd.DataFrame(({'lat': lat, 'lon': lon}))
header = ['lat','lon']
coords.to_csv('Coordinates_full.csv', sep = ',', columns = header)
```

### 5. UFO_Shape_Classification.py

Code for making shape predictions based on sightings features such as location etc. Modeling is far from exhaustive, and I may revisit this dataset in the future when/if my interest resurfaces.

### 6. UFO_Time_Series.py 

Code for visualizing sightings throughout the year. Contains recurrent neural network code for predicting sightings counts over time (time series). I used the model from <http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/>, but didn't work extensively on improving results.
