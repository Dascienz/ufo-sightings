#!/usr/bin/env python3
import os
import time
import datetime
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

#current directory
wdir = os.path.dirname(__file__)

#filenames
report_file = os.path.join(wdir,"data/nuforc_reports_07202018.csv")
geo_file = os.path.join(wdir,"data/lat-lon-coordinates.txt")

#reports data
data = pd.read_csv(report_file, sep=",", usecols=["Location"], encoding="ISO-8859-1")

def main():
	"""Main process for looking up decimal latitude and longitude coordinates for (str) locations."""
	
	#geolocator
	geolocator = Nominatim()
	locations = data["Location"].unique()
	location_dict = {}

	lines = 0
	with open(geo_file, "w", encoding="utf-8") as out_file:
		for idx, loc in enumerate(locations): 
			sleep(1.5) #Sleep for 1.5 seconds between each query.
			if idx == 0: 
				out_file.write("Index Location Latitude Longitude")
			try:
				geoLoc = geolocator.geocode(loc, timeout=60.0)
				out_file.write("\n{0}|{1}|{2}|{3}".format(idx, loc, geoLoc.latitude, geoLoc.longitude))
			except:
				out_file.write("\n{0}|{1}|{2}|{2}".format(idx,loc,"NaN"))
			lines += 1
			print("Processed {0} locations out of {1}\r".format(lines,len(locations)), end="")

			
if __name__=="__main__":
	if os.path.isfile(geo_file):
		print("\nGeographic coordinates file already exists at /{}.".format(geo_file))
	else:
		main()