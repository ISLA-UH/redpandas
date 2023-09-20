"""
Figure parameters class
"""


class FigureParameters:
    """
    Class contains figure and text sizes
    """
    def __init__(self):
        self.scale = 1.25*1080/8
        self.figure_size_x = int(1920/self.scale)
        self.figure_size_y = int(1080/self.scale)
        self.text_size = int(2.9*1080/self.scale)
        self.text_size_minor_yaxis = 8
        # Colormap/
        self.color_map = "inferno"  # 'hot_r'  # 'afmhot_r' #colormap for plotting
