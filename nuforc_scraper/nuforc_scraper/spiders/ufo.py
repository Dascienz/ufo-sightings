# -*- coding: utf-8 -*-

import re
import scrapy
from bs4 import BeautifulSoup
from scrapy.http import Request
import urllib.parse as urlparse

class UfoSpider(scrapy.Spider):
	name = 'ufo'
	allowed_domains = ['nuforc.org']
	shape_hrefs = ['ndxs.html','ndxschanged.html','ndxsChanging.html',
	'ndxsChevron.html','ndxsCigar.html','ndxsCircle.html','ndxsCone.html',
	'ndxsCrescent.html','ndxsCross.html','ndxsCylinder.html','ndxsDelta.html',
	'ndxsDiamond.html','ndxsDisk.html','ndxsDome.html','ndxsEgg.html',
	'ndxsFireball.html','ndxsFlare.html','ndxsFlash.html','ndxsFormation.html',
	'ndxsHexagon.html','ndxsLight.html','ndxsOther.html','ndxsOval.html',
	'ndxspyramid.html','ndxsRectangle.html','ndxsRound.html','ndxsSphere.html',
	'ndxsTeardrop.html','ndxsTriangle.html','ndxsTRIANGULAR.html','ndxsUnknown.html']
	start_urls = ['http://nuforc.org/webreports/%s' % (href) for href in shape_hrefs]
	
	def parse(self, response):
		event_links = response.xpath('//td//a/@href').extract()
		for link in event_links:
			url = urlparse.urljoin(response.url, link)
			request = scrapy.Request(url, callback=self.parse_report)
			yield request
	
	def parse_report(self, response):
		
		#collect meaningful tags from page
		soup = BeautifulSoup(str(response.body), 'lxml')
		report = [tag.find('font').text for tag in soup.find_all('td')]
		
		#parse details
		details = report[0].replace(' :',':')
		Occurred = re.findall('Occurred:(.*?)Reported', details)[0].strip().strip(',')
		Reported = re.findall('Reported:(.*?)Posted', details)[0].strip().strip(',')
		Posted = re.findall('Posted:(.*?)Location', details)[0].strip().strip(',')
		Location = re.findall('Location:(.*?)Shape', details)[0].strip().strip(',')
		Shape = re.findall('Shape:(.*?)Duration', details)[0].strip().strip(',')
		Duration = re.findall('Duration:(.*?)$', details)[0].strip().strip(',')
		Report = report[1].strip().strip(',')
		
		yield {'Occurred':Occurred,'Reported':Reported,'Posted':Posted,'Location':Location,'Shape':Shape,'Duration':Duration,'Report':Report}