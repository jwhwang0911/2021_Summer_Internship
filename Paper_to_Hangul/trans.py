import argparse
import shutil
import tika
tika.initVM()
from tika import parser

# python3 trans.py -i /rec/pvr1/5_SigAsia_2021_Monte_Carlo_Denoising_low_res.pdf

pa = argparse.ArgumentParser()
pa.add_argument("--pdfdir","-i", required=True)
args, unknown = pa.parse_known_args()

if __name__ == "__main__":
    parsed = parser.from_file(args.pdfdir)
    txt = open('output.txt','w',encoding='utf-8')
    print(parsed['content'], file=txt)
    txt.close
    pass