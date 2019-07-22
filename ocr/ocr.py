import tempfile
import subprocess

def text_from_image_file(image_name,lang):
  output_name = tempfile.mktemp()
  return_code = subprocess.call(['tesseract',image_name,output_name,'-l',lang],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  d = open(output_name+'.txt','r',encoding='utf-8')
  return d.read()


