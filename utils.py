def plot_curves(ax, x, curves, title, x_label, y_label, log=False):
    for y, label, ls in curves:
        ax.plot(x, y, label=label, linestyle=ls)

    ax.set_xscale("log" if log else "linear")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.legend()
    ax.grid()
