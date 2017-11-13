# UFO_Sightings
Scraped data from www.nuforc.org

This is a series of .py files for scraping, visualizing, and analyzing U.F.O. data from www.nuforc.org. Some .csv files are also provided. Files can be run as is in python 3.6.

## DATA FILES

### nuforcReports.csv
Unzip `Archive.zip` and import nuforcReports.csv into `ufoProcessing.py`. The script will output a file named `ufoData.csv` which contains structured U.F.O. sightings data which you can use for analysis and modeling. You're welcome to scrape your own files using `ufoSpider.py`, but obtaining geographic coordinates can be annoying since server requests can take a long time and you have the potential of getting kicked.

### locationData.csv
Contains latitude and longitude coordinates for sightings. These results were obtained using `geopy`.

### airportData.csv
Latitude and longitude coordinates for airports, heliports, and seaplane bases across the United States. Spreadsheet sliced from data obtained from <https://www.faa.gov/airports/airport_safety/airportdata_5010/>.

### militaryData.csv
Latitude and longitude coordinates for military bases across the United States. Obtained from <https://www.google.com/maps/d/viewer?mid=1hvB-oq9gE0H8gEwKJ4XHFOKaY5k&hl=en_US&ll=49.67042577190774%2C-117.29381349999994&z=2>

Download the KML file and convert it to CSV using software of your choice.
