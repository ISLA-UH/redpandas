"""
Plot coherence
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_psd_coh(psd_sig,
                 psd_ref,
                 coherence_sig_ref,
                 f_hz,
                 f_min_hz,
                 f_max_hz,
                 f_scale: str = "log",
                 sig_label: str = "PSD Sig",
                 ref_label: str = "PSD Ref",
                 psd_label: str = "PSD (bits)",
                 coh_label: str = "Coherence",
                 f_label: str = "Frequency (Hz)",
                 fig_title: str = "Power spectral density and coherence",
                 show_figure: bool = True) -> Figure:
    """
    Plot coherence and power spectral density

    :param psd_sig: Power spectral density of signal
    :param psd_ref: Power spectral density of reference signal
    :param coherence_sig_ref:  magnitude squared coherence of x and y
    :param f_hz: sample frequencies of PSD
    :param f_min_hz: minimum frequency to plot in Hz (x min limit)
    :param f_max_hz: maximum frequency to plot in Hz (x max limit)
    :param f_scale: scale of x-axis. One of {"linear", "log", "symlog", "logit"}. Default is "log"
    :param sig_label: label for signal. Default is "PSD Sig"
    :param ref_label: label for reference. Default is "PSD Ref"
    :param psd_label: label for PSD. Default is "PSD (bits)"
    :param coh_label: label for coherence. Default is "Coherence"
    :param f_label: x-axis label. Default is "Frequency (Hz)"
    :param fig_title: title of figure. Default is "Power spectral density and coherence"
    :param show_figure: show figure is True. Default is True

    :return: matplotlib figure instance
    """
    # Plot PSDs
    fig1 = plt.figure()
    fig1.set_size_inches(8, 6)
    plt.clf()
    ax1 = plt.subplot(211)
    ax1.plot(f_hz, psd_ref, 'r-', linewidth=2, label=ref_label)
    ax1.plot(f_hz, psd_sig, 'k-', label=sig_label)
    ax1.set_xscale(f_scale)
    ax1.legend()
    ax1.set_xlim([f_min_hz, f_max_hz])
    ax1.set_ylim([-16, 1])
    ax1.set_ylabel(psd_label)
    ax1.grid('on', which='both')
    ax1.set_title(fig_title)

    ax2 = plt.subplot(212)
    ax2.plot(f_hz, coherence_sig_ref, 'k-')
    ax2.set_xscale(f_scale)
    ax2.set_xlim([f_min_hz, f_max_hz])
    ax1.set_ylim([-16, 1])
    ax2.set_xlabel(f_label)
    ax2.set_ylabel(coh_label)
    ax2.grid('on', which='both')

    if show_figure is True:
        plt.show()

    return fig1


def plot_response_scatter(h_magnitude,
                          h_phase_deg,
                          color_guide,
                          f_hz,
                          f_min_hz,
                          f_max_hz,
                          f_scale: str = 'log',
                          fig_title: str = 'Response only valid at high coherence',
                          show_figure: bool = True) -> Figure:
    """
    Plot coherence response

    :param h_magnitude: magnitude, for example, absolute magnitude of response (which is power spectral density /
        cross-power spectral density)
    :param h_phase_deg: coherence phase degrees
    :param color_guide: parameters color guide, for example, magnitude squared coherence of x and y
    :param f_hz: frequency of coherence in Hz
    :param f_min_hz: minimum frequency to plot in Hz (x min limit)
    :param f_max_hz: maximum frequency to plot in Hz (x max limit)
    :param f_scale: scale of x axis. One of {"linear", "log", "symlog", "logit"}. Default is "log"
    :param fig_title: title of figure
    :param show_figure: show figure is True. Default is True

    :return: matplotlib figure instance
    """
    # plot magnitude and coherence
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    ax1 = plt.subplot(211)
    im1 = ax1.scatter(x=f_hz, y=h_magnitude, c=color_guide, marker='o')
    ax1.set_xscale(f_scale)
    ax1.set_xlim([f_min_hz, f_max_hz])
    ax1.grid('on', which='both')
    hc = fig.colorbar(im1)
    hc.set_label('Coherence')
    ax1.set_ylabel('Magnitude ')
    ax1.set_title(fig_title)

    ax2 = plt.subplot(212)
    im2 = ax2.scatter(x=f_hz, y=h_phase_deg, c=color_guide, marker='o')
    ax2.set_xscale(f_scale)
    ax2.set_xlim([f_min_hz, f_max_hz])
    ax2.grid('on', which='both')

    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [deg]')
    hc = plt.colorbar(im2)
    hc.set_label('Coherence')

    if show_figure is True:
        plt.show()

    return fig
