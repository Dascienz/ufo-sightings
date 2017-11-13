# UFO_Sightings
Scraped data from www.nuforc.org

This is a series of .py files for scraping, visualizing, and analyzing U.F.O. data from www.nuforc.org. Some .csv files are also provided. Files can be run as is in python 3.6.

## DATA FILES

### nuforcReports.csv
To obtain the scraped results .csv file run the following commands from your terminal:
1. `cd /path/to/files/directory`
2. `zip -F splituforeports.zip --out uforeports.zip`
3. `unzip uforeports.zip`

You should have a file named uforeports.csv which contains raw U.F.O. sightings entries scraped from nuforc.org. You're welcome to scrape your own files using `UFO_Spider.py`.

### locationData.csv
Contains latitude and longitude coordinates for sightings. These results were obtained using geopy.

### airportData.csv
Latitude and longitude coordinates for airports, heliports, and seaplane bases across the United States. Spreadsheet sliced from data obtained from <https://www.faa.gov/airports/airport_safety/airportdata_5010/>.

### militaryData.csv
Latitude and longitude coordinates for military bases across the United States. Obtained from <https://www.google.com/maps/d/viewer?mid=1hvB-oq9gE0H8gEwKJ4XHFOKaY5k&hl=en_US&ll=49.67042577190774%2C-117.29381349999994&z=2>

Download the KML file and convert it to CSV using software of your choice.
