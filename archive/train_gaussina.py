import numpy as np

blue = [[108,65,38],[129,72,41],[126,76,40],[121,75,44],[118,68,38],[111,66,45],[97,65,42],[118,71,40],[125,68,42],[141,73,36],[130,79,46],[154,85,45]]
blue = np.array(blue)
print blue
print np.mean(blue,axis=0)

print np.cov(blue.T)


