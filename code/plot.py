
# from matplotlib import pyplot as plt
# plt.rcParams['text.usetex'] = True
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"

# Helper function plotting the PDFs


def plot_results(exp_name, norm_data, anom_data):
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('GTKAgg')
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    import seaborn as sns
    sns.set(style="white", color_codes=True)
    sns.set_context("paper")
    import numpy as np
    fig = plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.rcParams['lines.linewidth'] = 3
    ax = fig.add_subplot(111)
    ax.plot(norm_data, label='Normal PDFs')
    plt.plot(anom_data, label='Anomalous PDFs')
    x_max = np.array([np.array(anom_data).max(), np.array(norm_data).max()])
    y_max = np.array([np.array(norm_data).max(), np.array(anom_data).max()])
    # y_min = np.array([np.array(norm_data).min(), np.array(anom_data).min()])
    plt.xlim(0, x_max.max())
    plt.ylim(-1, y_max.max())
    filename = exp_name + '_thresh_pdfs.png'
    plt.xlabel('Probability Density Function Index', fontsize=27)
    plt.ylabel('Probability Density Function Values', fontsize=27)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.axhline(y=2.3, linewidth=4, color='r', ls='-.',
                label='Threshold', visible=True, solid_capstyle='round')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, prop={'size': 15},
               ncol=2, mode="expand", borderaxespad=0.)
    # plt.show()
    plt.savefig(filename, orientation='landscape', dpi=80, papertype='legal', format='png', quality=95,
                facecolor='w', edgecolor='w', bbox_inches='tight', pad_inches=0.5, transparent=True)


# if __name__ == '__main__':
#     freeze_support()
#     exp_name = ['fifo', 'sporadic', 'full', 'hilrf']

#     with Pool(processes=4) as pool:
#         pool.map(eval_model, exp_name)
