#!/usr/bin/env python3
import os
import re
import sys
import datetime
import numpy as np
import pandas as pd

#current directory
wdir = os.path.dirname(__file__)

#filenames
report_file = os.path.join(wdir,"data/nuforc_reports_07202018.csv")
geo_file = os.path.join(wdir,"data/lat-lon-coordinates.txt")

#geo-coordinates and reports dataframes
geo = pd.read_csv(geo_file,sep="|",index_col=0,encoding="ISO-8859-1")
data = pd.read_csv(report_file, sep=",",encoding="ISO-8859-1")

# --------------------------------------------------------------
# FUNCTIONS FOR CLEANING SCRAPED UFO SIGHTINGS REPORTS
# --------------------------------------------------------------

#CLEANING DATE AND TIME COLUMNS
def date_time_filter(series):
	"""Function for filtering datetime values.
	----- Args:
			series: pandas.Series
	----- Returns:
			mapped pandas.Series
	"""
	
	pattern = "(\d+\/\d+\/\d+ \d+:\d+)|(\d+\/\d+\/\d+)|(\d+:\d+)|(\d+ \d+)|\(Entered as:(.*?)\)" #grab datetimes
	filter_map = {x:list(filter(None,re.findall(pattern,str(x))[0]))[0] for x in series.unique()}
	return series.map(filter_map)

	
def time_picker(series):
	"""Function for picking out %H-%m-%s time-strings from datetimes.
	----- Args:
			series: pandas.Series
	----- Returns:
			mapped pandas.Series
	"""
	
	times = {}
	for t in series.unique():
		try:
			times[t] = re.findall("\d+:\d+",t)[0]
		except:
			times[t] = np.nan
	return series.map(times)


def str_to_datetime(series):
    """Function for converting (str) datetimes to (numpy.datetime64) datetimes.
	----- Args:
			series: pandas.Series
	----- Returns:
			mapped pandas.Series
	"""
	
    datetimes = {t:pd.to_datetime(t,errors="coerce") for t in series.unique()}
    return series.map(datetimes)

	
#CLEANING LOCATION COLUMNS
def str_location_cleaner(series):
	"""Function for mapping cleaned location strings.
	----- Args:
			series: (str) pandas.Series
	----- Returns:
			cleaned (str) pandas.Series
	"""

	def clean_location(loc):
		"""Function for cleaning location string values.
		----- Args:
				loc: (str) location
		----- Returns:
				cleaned (str) location
		"""
		
		descriptors = ["north of","south of","east of","west of",
					   "outside of","between","above","near","SW part",
					   "NW part","SE part","NE part"]
		for r in ["?",", ."]:
			loc = loc.replace(r,"")  
		if ("(" not in loc) or (")" not in loc):
			return loc
		else:
			pattern = "\(.*?\)"
			items = re.findall(pattern, loc)
			for i, item in enumerate(items):
				if item == "()":
					loc = loc.replace(item,"")
				if (len(item.split()) >= 3) or any(s in item for s in descriptors):
					loc = loc.replace(item,"")
			return re.sub(" +"," ", loc.replace(" ,",","))
	
	locations = {x:clean_location(x) for x in series.unique()} 
	return series.map(locations)
	
	
#CLEANING DURATION COLUMN
def str_duration_cleaner(series):
	"""Function for mapping cleaned (str) duration values.
	----- Args:
			series: pandas.Series
	----- Returns:
			mapped pandas.Series
	"""

	def clean_duration(s):
		"""Function for cleaning location string values.
		----- Args:
				loc: (str) location
		----- Returns:
				cleaned (str) location
		"""
		
		s = s.replace("few", "3.5").replace("1/2","0.5") #use convention that a few seconds translates to 2 - 5 seconds.
		numericList = ["(\d+\.\d+)","(\d+)"]
		unitList = ["se[cs]", "secon[ds]","mi[ns]", "mi[mn]ut[es]","h[rs]", "hou[rs]"]
		unitConversion = {'s':1/60, 'm':1, 'h':60, '6':60}
		try:
			match_1 = re.search(re.compile("|".join(str(x) for x in numericList)), s).group(0)
			match_2 = re.search(re.compile("|".join(str(x) for x in unitList)), s.lower()).group(0)[0]
			s = float(match_1)*unitConversion[match_2]
		except:
			s = np.nan
		return s
	
	durations = {x:clean_duration(x) for x in series.unique()}
	return series.map(durations)


def capitalizer(series):
	"""Function for capitalizing the first letter in each word.
	----- Args:
			series: pandas.Series
	----- Returns:
			mapped pandas.Series
	"""
	
	strings = {x:x[0].upper()+x[1:].lower() for x in series.unique()}
	return series.map(strings)


def all_shapes_in_report(reportSeries, shapeSeries):
	"""Function for correcting unknown and other shape categories
	based on entered witness descriptions. Uses shape label detection
	within each report to predict the shape of a given ufo.
	----- Args:
			reportSeries: pandas.Series
			shapeSeries: pandas.Series
	----- Returns:
			list() of revised shape categories
	"""

	def shape_category_counter(report, shape_categories):
		"""Function counts the occurrence of shape category terms in
		each written witness report.
		----- Args:
				report: (str) description
				shape_categories: list() unique shape labels
		----- Returns:
				list() category-keys"""
				
		report = report.lower().split()
		cat_count = dict(zip(list(shape_categories),len(shape_categories)*[0]))
		for word in report:
			if word in shape_categories:
				cat_count[word] += 1
		cat_count = {k[0].upper()+k[1:]:v for k,v in cat_count.items() if v}
		return list(cat_count.keys())
	
	unknown_categories = ["Unknown","Other"]
	additional_categories = ["orb"] #assert lower-case when adding shapes
	shape_categories = [x.lower() for x in shapeSeries.unique() if x not in unknown_categories] + additional_categories
	
	#every report is unique, won't use a look-up dict()
	#form pandas.Series containing list() values of shapes
	z1 = reportSeries.apply(lambda report: shape_category_counter(report, shape_categories))

	z2 = []
	for x, y in zip(shapeSeries, z1):
		#note y is a list()
		if (x in unknown_categories) and (y != []):
			z2.append(y)
		elif (x in unknown_categories) and (y == []):
			z2.append(y + [x])
		elif (x not in unknown_categories) and (x not in y):
			z2.append(y + [x])
		else:
			z2.append(y)
	return z2

	
def export_pickle(data):
	"""Function for exporting cleaned pandas.DataFrame to pickle."""
	
	split = report_file.split("/")
	pickle_file = split[0] + "/cleaned_" + split[-1].replace("csv","pkl")	
	data.to_pickle(pickle_file)
	
	
#MAIN FUNCTION
def clean(data):
	"""Main function that applies cleaning functions to raw data columns."""
	
	#DROP NULL AND RESET INDEX
	data = data.dropna().reset_index(drop=True)
	
	#CLEAN DATE AND TIME COLUMNS
	print("\nCleaning date and time data...")
	data["Occurred"] = date_time_filter(data["Occurred"])
	
	data["Sighting Time"] = time_picker(data["Occurred"])
	data["Report Time"] = time_picker(data["Reported"])
	
	data["Occurred"] = str_to_datetime(data["Occurred"])
	data["Reported"] = str_to_datetime(data["Reported"])
	data["Posted"] = str_to_datetime(data["Posted"])
	
	#CLEAN LOCATION COLUMN
	print("\nCleaning location data...")
	data["Location"] = str_location_cleaner(data["Location"])
	
	lat_dict = dict(zip(geo["Location"],geo["Latitude"]))
	lon_dict = dict(zip(geo["Location"],geo["Longitude"]))
	
	data["Latitude"] = data["Location"].map(lat_dict)
	data["Longitude"] = data["Location"].map(lon_dict)
	data = data.replace({"NaN":np.nan}, regex=True)
	
	#CLEAN SHAPE COLUMN
	print("\nCleaning shape data...")
	data["Shape"] = capitalizer(data["Shape"])
	
	shape_replace_dict = {"Changed":"Changing","Flare":"Fireball","Triangular":"Triangle"}
	data["Shape"] = data["Shape"].replace(shape_replace_dict,regex=True)
	data["Shape Categories Revised"] = all_shapes_in_report(data["Report"],data["Shape"])
	
	export_pickle(data)
	#end


if __name__=="__main__":
	clean(data)