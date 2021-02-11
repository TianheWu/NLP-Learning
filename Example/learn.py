import numpy as np
from pandas import DataFrame

data = DataFrame(np.arange(10,26).reshape((4, 4)))

aList = [123, 'xyz', 'zara', 'abc', 123];
bList = [2009, 'manni'];
aList.extend(bList)

print(aList)