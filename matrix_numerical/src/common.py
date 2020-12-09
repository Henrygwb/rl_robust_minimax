env_list = ["Match_Pennies", "As_Match_Pennies", "CC", "NCNC"]


def convex_concave(x, y):
    return x * x - y * y


def non_convex_non_concave(x, y):
    return x * x * y * y - x * y