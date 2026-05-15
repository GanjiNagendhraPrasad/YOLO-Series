import argparse

def sharuk(a,b):
    print(a + b)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--value_of_a" , type=int , default=30,help="Pass value a")
    args.add_argument("--value_of_b", type=int, default=80, help="Pass value b")

    opt = args.parse_args()
    sharuk(opt.value_of_a , opt.value_of_b)
