"""Page/frame preview with actual extracted layout rectangles."""
import io
from pathlib import Path
import subprocess
import tempfile


def render_page(name,content,number,boxes=None):
    from PIL import Image, ImageDraw
    with tempfile.TemporaryDirectory(prefix='bank-preview-') as temp:
        if Path(name).suffix.lower()=='.pdf':
            source=Path(temp)/'original.pdf'
            source.write_bytes(content)
            prefix=Path(temp)/'page'
            subprocess.run(['pdftoppm','-f',str(number),'-l',str(number),'-singlefile','-scale-to','1800','-png',str(source),str(prefix)],check=True,capture_output=True,timeout=30)
            with Image.open(prefix.with_suffix('.png')) as source_image:
                image=source_image.convert('RGBA')
        elif Path(name).suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff'):
            with Image.open(io.BytesIO(content)) as source_image:
                source_image.seek(number-1)
                image=source_image.convert('RGBA')
        else:
            raise ValueError('Для текстовых файлов изображение страницы недоступно.')
        if boxes:
            overlay=Image.new('RGBA',image.size)
            draw=ImageDraw.Draw(overlay)
            for rect in boxes['rects']:
                coords=[rect[0]*image.width/boxes['width'],rect[1]*image.height/boxes['height'],
                        rect[2]*image.width/boxes['width'],rect[3]*image.height/boxes['height']]
                draw.rectangle(coords,fill=(255,208,0,55),outline=(225,150,0,200),width=2)
            image=Image.alpha_composite(image,overlay)
        output=io.BytesIO()
        image.convert('RGB').save(output,format='PNG')
        return output.getvalue()
