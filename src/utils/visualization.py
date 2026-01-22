import matplotlib.pyplot as plt

def plot_section(section, title):
    plt.imshow(section, cmap="seismic", aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.show()
