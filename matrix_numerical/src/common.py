env_list = ["Match_Pennies", "As_Match_Pennies", "CC", "As_CC", "NCNC"]


def convex_concave(x, y):
    return x * x - y * y


# def as_convex_concave(x, y):
#     return x * x - y * y - 2 * x

def as_convex_concave(x, y):
    return x * x - 2 * y * y - 2 * x * y - 6 * x


def non_convex_non_concave(x, y):
    return x * x * y * y - x * y