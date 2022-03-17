import math

__all__ = ["timestr"]


def timestr(sec, day=False):
    sec = round(sec)
    out = ""
    if day:
        days, sec = divmod(sec, 24 * 3600)
        out += f"{days} day, "

    hour, sec = divmod(sec, 3600)
    out += "{:02d}h ".format(hour)

    minutes, sec = divmod(sec, 60)
    out += "{:02d}m {:02d}s".format(minutes, sec)
    return out
