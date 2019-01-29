import matplotlib.pyplot as plt


class MultiplePlot:
    def __init__(self, size, _dimensions):
        self.ax = []
        self.images_num = 0
        self.fig = plt.figure(figsize=size)
        self.dimensions = _dimensions

    def add(self, image, title, _cmap='gray'):
        self.images_num += 1
        self.ax.append(self.fig.add_subplot(self.dimensions[0], self.dimensions[1], self.images_num))
        self.ax[-1].set_title(title)
        plt.imshow(image, cmap=_cmap)


    def show(self):
        plt.show()


def show(image, _cmap='gray'):
    plt.imshow(image, cmap=_cmap)
    plt.show()

