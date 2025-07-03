import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

class HandlerArrow(HandlerPatch):
    ''''
    Handler for arrows in legends.
    '''
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Adjust these to control arrow length
        start = (xdescent + width * 0.1, ydescent + height / 2)
        end   = (xdescent + width * 0.9, ydescent + height / 2)
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='->', mutation_scale=15,
            color=orig_handle.get_edgecolor() or orig_handle.get_facecolor(),
            transform=trans
        )
        return [arrow]
    
class HandlerEllipse(HandlerPatch):
    '''
    handler for ellipses in legends.    
    '''
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        ellipse = Ellipse(
            (xdescent + width / 2, ydescent + height / 2),
            width * 0.8, height * 0.5,  # control relative size
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            lw=orig_handle.get_linewidth(),
            transform=trans
        )
        return [ellipse]



def downsample_scatter_plot(samples, 
                            downsample=1000,
                            seed=None, 
                            ax=None,
                            plot_pair= None,
                            **kwargs):
    """
    Plot a scatter plot of the samples and scores with downsampling.

    Parameters:
    samples (numpy array): The samples to plot.
    downsample (int): The number of points to downsample to.
    ax (matplotlib axis): The axis to plot on.
    seed (int): The random seed for downsampling.
    **kwargs: Additional keyword arguments for the scatter function.

    Returns:
    None
    """
    if ax is None:
        fig, ax = plt.subplots()
    if seed is not None:
        np.random.seed(seed)
    if len(samples) > downsample:
        indices = np.random.choice(len(samples), downsample, replace=False)
        samples = samples[indices]
    if plot_pair is None:
        plot_pair = [0, 1]
    ax.scatter(samples[:, plot_pair[0]], samples[:, plot_pair[1]], **kwargs)

    return ax


def plot_vector_field(
    x_range,
    y_range,
    vector_field,
    ax=None,
    num_points=20,
    plot_pair=None,
    norm_vector=False,
    **kwargs,
):
    """
    Plot a vector field.

    Parameters:
    x_range (tuple): The range of x values.
    y_range (tuple): The range of y values.
    vectors (array-like): The vector field to plot.
    ax (matplotlib axis): The axis to plot on.
    num_points (int): The number of points to plot.
    color (str): The color of the arrows.
    alpha (float): The transparency of the arrows.
    **kwargs: Additional keyword arguments for the quiver function.

    Returns:
    None
    """
    if ax is None:
        fig, ax = plt.subplots()

    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    input = np.array([X.flatten(), Y.flatten()]).T
    vectors = vector_field(input)
    if plot_pair is None:
        plot_pair = [0, 1]
    vectors = vectors[:, plot_pair]

    if norm_vector:
        norm = np.linalg.norm(vectors, axis=1)
        vectors = vectors / norm[:, np.newaxis]

    U = vectors[:, 0]
    V = vectors[:, 1]

    ax.quiver(X, Y, U, V, **kwargs)

    return ax


def plot_contours(fisher, pos,  nstd=1., ax=None, **kwargs):
  """
  Plot 2D parameter contours given a Hessian matrix of the likelihood, copied from jax_cosmo.
  """
  
  def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

  mat = fisher
  cov = np.linalg.inv(mat)
  sigma_marg = lambda i: np.sqrt(cov[i, i])

  if ax is None:
      ax = plt.gca()

  vals, vecs = eigsorted(cov)
  theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

  # Width and height are "full" widths, not radius
  width, height = 2 * nstd * np.sqrt(vals)
  ellip = Ellipse(xy=pos, width=width,
                  height=height, angle=theta, **kwargs)

  ax.add_artist(ellip)
  return ax