# -*- coding: utf-8 -*-
"""
@author: Dascienz
"""
import scrapy
from scrapy.http import Request
import urllib.parse as urlparse

shape_list = ['Unspecified','changed','Changing','Chevron','Cigar','Circle','Cone','Crescent',
                  'Cross','Cylinder','Delta','Diamond','Disk','Dome','Egg','Fireball','Flare','Flash',
                  'Formation','Hexagon','Light','Other','Oval','pyramid','Rectangle','Round','Sphere',
                  'Teardrop','Triangle','TRIANGULAR','Unknown']
    
class ufoSpider(scrapy.Spider):
    name = 'ufo'
    allowed_domains = ['nuforc.org']
    start_urls = ['http://www.nuforc.org/webreports/ndxs%s.html' % shape_list[i] for i in range(len(shape_list))]

    def parse(self, response):
        """Collect all links from each url listed in start_urls."""
        links = response.xpath('//a/@href').extract()
        for link in links:
            url = urlparse.urljoin(response.url, link)
            yield scrapy.Request(url, callback = self.parse_report)
    
    def parse_report(self, response):
        """Yield each witness report."""
        description = response.xpath('//table//font//text()').extract()
        yield {'description': description}
