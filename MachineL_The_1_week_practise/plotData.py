import matplotlib.pyplot as plt

def plotData(x, y):
    # plt.ion()
    # plt.figure()
    plt.plot(x, y, 'x')
    # plt.axis([4,24,-5,25])
    # plt.xlabel("Population of City in 10,000s") # setting the x label as population
    # plt.ylabel("Profit in $10,000s") # setting the y label
    plt.show()

listx = [1,2,3]
listy = [1,2,3]
plotData(listx, listy)