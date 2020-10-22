from pylab import imshow, plot, show


def plotPoints(extremas=None, imgarr=None):
    first = extremas[0][1]

    x = []
    y = []

    for point in first:
        x.append(point[1])
        y.append(point[0])

    imshow(imgarr)
    plot(x, y, 'r.')
    show()
