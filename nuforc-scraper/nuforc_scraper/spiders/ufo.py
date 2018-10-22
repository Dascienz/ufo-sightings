# -*- coding: utf-8 -*-
import re
import scrapy
from bs4 import BeautifulSoup
from scrapy.http import Request


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
    start_urls = ['http://nuforc.org/webreports/{}'.format(href) for href in shape_hrefs]

    def parse(self, response):
        event_links = response.xpath('//td//a/@href').extract()
        for link in event_links:
            yield scrapy.Request(response.urljoin(link),callback=self.parse_report)
            
            
    def parse_report(self, response):
        #SCRAPY FIELD-DICT
        item = scrapy.Field()

        #FORM BEAUTIFUL SOUP OBJECT, GATHER FONT TAGS
        soup = BeautifulSoup(response.body,'lxml')
        report = [td.find('font').get_text() for td in soup.find_all('td')]

        #PARSE THROUGH DETAILS
        details = report[0].replace(' :',':')
        item['Occurred'] = re.findall('Occurred:(.*?)Reported',details)[0].strip().strip(',')
        item['Reported'] = re.findall('Reported:(.*?)Posted',details)[0].strip().strip(',')
        item['Posted'] = re.findall('Posted:(.*?)Location',details)[0].strip().strip(',')
        item['Location'] = re.findall('Location:(.*?)Shape',details)[0].strip().strip(',')
        item['Shape'] = re.findall('Shape:(.*?)Duration',details)[0].strip().strip(',')
        item['Duration'] = re.findall('Duration:(.*?)$',details)[0].strip().strip(',')
        item['Report'] = report[1].strip().replace('Summary:','').strip(',')
        yield item
        