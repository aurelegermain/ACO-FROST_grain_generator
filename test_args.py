# %%
import argparse

parser = argparse.ArgumentParser()

#-db DATABSE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-g", "--gfn", help="GFN-xTB method to use", default="2")
parser.add_argument("-size", "--size", help="Size", type=int, default="0")
parser.add_argument("-md", "--md", help="Molecular Dynamics Interval", type=int, default="0")

args = parser.parse_args()

print( "GFN {} size {} MD {}".format(
        args.gfn,
        args.size,
        args.md
        ))

gfn = args.gfn
size = args.size
md = args.md


# %%
