import os

def compare_dates(date_1, date_2):
    """
    A function used to compare 2 dates: date_1 and date_2.
    Args:
        date_1 (str): a strictly formated date, eg. 2023/03/26 (Sun) 22:45.
        date_2 (str): same type as date_1.
    Returns:
        comparison (int): if `comparison==-1`, date_1 is earlier than date_2; if `comparison==0`, date_1 is the same as date_2; if `comparison==+1`, date_1 is later than late_2.
    """
    # Extract year, month, day, hour, minute from date_1
    year_1 = int(date_1[0:4])
    month_1 = int(date_1[5:7])
    day_1 = int(date_1[8:10])
    hour_1 = int(date_1[17:19])
    minute_1 = int(date_1[20:22])

    # Extract year, month, day, hour, minute from date_2
    year_2 = int(date_2[0:4])
    month_2 = int(date_2[5:7])
    day_2 = int(date_2[8:10])
    hour_2 = int(date_2[17:19])
    minute_2 = int(date_2[20:22])

    # Compare years
    if year_1 < year_2:
        return -1
    elif year_1 > year_2:
        return 1
    
    # Compare months
    if month_1 < month_2:
        return -1
    elif month_1 > month_2:
        return 1

    # Compare days
    if day_1 < day_2:
        return -1
    elif day_1 > day_2:
        return 1

    # Compare hours
    if hour_1 < hour_2:
        return -1
    elif hour_1 > hour_2:
        return 1

    # Compare minutes
    if minute_1 < minute_2:
        return -1
    elif minute_1 > minute_2:
        return 1

    # If all are equal, return 0
    return 0


print(compare_dates("2028/03/26 (Sun) 22:45", "2024/08/27 (Sun) 10:45"))
