from PIL import Image
from PIL import ImageStat
import os

path = "imgs/"
masks = [Image.open("mask-1.png"), Image.open("mask-2.png"), Image.open("mask-3.png")]

l = os.listdir(path) # returns list
for x in l:
	im = Image.open(path + x)
	max_degree = 0
	max_val = 0
	for r in xrange(0, 360, 4):
		for m in masks:
			val = ImageStat.Stat(im.rotate(r), m).mean[0]
			if val > max_val:
				max_degree = r
				max_val = val
				
	Image.open("png/" + x).rotate(max_degree, Image.BILINEAR).save("rotated/" + x)
	im.rotate(max_degree, Image.BILINEAR).save("prob_rotated/" + x)