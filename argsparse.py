import argparse

parser = argparse.ArgumentParser(description="Simple argument parser example")

parser.add_argument("name",type=str,help="Your Name")
parser.add_argument("math",type=int,help="Maths mark")
parser.add_argument("physics",type=int,help="Physics mark")

args = parser.parse_args()

print(f"Hi {args.name}, your maths mark is {args.math} and physics mark is {args.physics}")