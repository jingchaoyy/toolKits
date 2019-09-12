"""
Created on  2019-09-12
@author: Jingchao Yang
"""
"""
Created on  2019-09-09
@author: jc
"""
import os
import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pandas as pd


def add_text_to_img(dir, text, font, size, out_dir):
    """

    :param dir:
    :param text:
    :param font:
    :param size:
    :param out_dir:
    :return:
    """
    img = Image.open(dir)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype(font, size)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0, 0), text, (0, 0, 0), font=font)
    img.save(out_dir)


path = '/image_sample/'
dir = glob.glob(path + '*.png')
rng = pd.date_range(start='5/5/2019', periods=72, freq='H')

for f in range(len(dir)):
    head, tail = os.path.split(dir[f])
    fname = tail
    # ntext = fname.split('.')[0]
    # ntext = int(ntext) % 24
    ntext = str(rng[f])
    ntext = ntext.split(' ')
    ntext = ntext[0] + '\n' + ntext[1]
    add_text_to_img(dir[f], str(ntext), "/fast_99/fast99.ttf", 100, path + 'processed/' + fname)
