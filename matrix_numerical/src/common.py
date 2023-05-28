env_list = ["Match_Pennies", "CC", "As_CC", "NCNC", "CC_1", "As_CC_1", "NCNC_1"]

def convex_concave(x, y):
    return x * x - y * y - 2 * x

def as_convex_concave(x, y):
    return x * x - 2 * y * y - 2 * x * y - 6 * x

def non_convex_non_concave(x, y):
    return x * x * y * y - x * y

# NE (-4, -1)
def convex_concave_complex(x, y):
    return x * x + 2 * x * y - 4 * y * y + 10 * x - 6
# NE (-4, -4)
def as_convex_concave_complex(x, y):
    return x * x + 4 * x * y - 2 * y * y + 24 * x

# NE (6, 0)
def non_convex_non_concave_complex(x, y):
    return x * x * x - 9 * x * x - 2 * y * y * x * x * x