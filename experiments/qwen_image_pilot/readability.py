"""Create a diagnostic contact sheet; never changes evaluation image bytes."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
names=['2971a400310d571207764c4a064625dfbdf27413__clean.png','eefe66428206c7432df71d60213e23be2974be0b__wrong__composite.png']
pairs=[(Image.open(a.root/'images'/n).convert('RGB'),Image.open(a.root/'processed'/n).convert('RGB')) for n in names]
width=max(im.width for pair in pairs for im in pair)+16
row_height=300+38
sheet=Image.new('RGB',(width*2,row_height*6),'#e8e8e8');d=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=16)
for pair_index,pair in enumerate(pairs):
    for part,frac in enumerate((0,.5,1)):
        row=pair_index*3+part
        for col,im in enumerate(pair):
            y=round((im.height-300)*frac)
            d.text((col*width+8,row*row_height+8),f'{"Rendered" if col==0 else "Processor pixels"} / {"target" if pair_index==0 else "composite"} / {("top","middle","bottom")[part]}',fill='black',font=font)
            sheet.paste(im.crop((0,y,im.width,y+300)),(col*width+8,row*row_height+32))
sheet.save(a.root/'readability-contact-sheet.png')
