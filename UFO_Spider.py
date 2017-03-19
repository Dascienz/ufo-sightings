# -*- coding: utf-8 -*-
"""
@author: Dascienz
"""
import scrapy
import urlparse
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.shell import inspect_response

# relative_urls = response.xpath('//table//a').css('a::attr(href)').extract()
# index_titles = response.xpath('//table//a/text()').extract()

# Acquired shape_list from performing a response.xpath().extract()
# on http://nuforc.org/webreports/ndxshape.html

shape_list = ['Unspecified','changed','Changing','Chevron','Cigar',
              'Circle','Cone','Crescent','Cross','Cylinder','Delta',
              'Diamond','Disk','Dome','Egg','Fireball','Flare','Flash',
              'Formation','Hexagon','Light','Other','Oval','pyramid',
              'Rectangle','Round','Sphere','Teardrop','Triangle',
              'TRIANGULAR','Unknown']
    
class UfoSpider(scrapy.Spider):
    name = 'ufo'
    allowed_domains = ['nuforc.org']

    start_urls = ['http://www.nuforc.org/webreports/ndxs%s.html' % shape_list[i] for i in range(len(shape_list))]
    
    rules = (Rule(LinkExtractor(allow=(), ),
             callback='parse_index', follow = True),)
    
    download_delay = 0.25 #Be courteous with this parameter!

    def parse(self, response):
        #inspect_response(response, self)
        for href in response.xpath('//a/@href'):
            url = href.extract()
            url = urlparse.urljoin(response.url, url)
            yield scrapy.Request(url, callback = self.parse_item)
            
    def parse_index(self, response):
        #inspect_response(response,self)
        for href in response.xpath('//a/@href'):
            url = href.extract()
            url = urlparse.urljoin(response.url, url)
            yield scrapy.Request(url, callback = self.parse_item)
    
    def parse_item(self, response):
        description = response.xpath('//table//font//text()').extract()
        yield {'description': description}

# Run from the Terminal command line.
# 1.) pip install scrapy
# 2.) scrapy crawl UFO_Spider -o uforeports.csv

