import os
import csv
import matplotlib.pyplot as plt
import csv

def _log_passed_arguments(logging, arguments):
    for key, value in arguments.items():
        if key is not 'self':
            logging.info('%s: %s', key, value)

def _get_files_in_folder(input_folder, ext=".xml.gz", start_file=None, end_file=None):
        currentDir = os.getcwd() if input_folder == '/' else input_folder
        start = os.path.join(currentDir, start_file) if start_file is not None else None
        end = os.path.join(currentDir, end_file) if end_file is not None else None
        files = []
        for file in os.listdir(currentDir):
            if file.endswith(ext):
                files.append(os.path.join(currentDir, file))        
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
        files = files[files.index(start) if start is not None else None : files.index(end) if end is not None else None ]
        return files

def _draw_plots():
    gsim = []
    nsim = []

    gp = []
    np = []

    gr = []
    nr = []

    gf = []
    nf = []

    with open('../results/sim_score_results.csv','r') as gold, open('../results/sim_score_ncbo_results.csv', 'r') as ncbo:
        gold_plots = csv.reader(gold, delimiter=',')
        ncbo_plots = csv.reader(ncbo, delimiter=',')
        next(gold_plots, None)
        next(ncbo_plots, None)
        for row in gold_plots:
            gsim.append(float(row[0]))
            gp.append(float(row[4]))
            gr.append(float(row[5]))
            gf.append(float(row[3]))
        for row in ncbo_plots:
            nsim.append(float(row[0]))
            np.append(float(row[4]))
            nr.append(float(row[5]))
            nf.append(float(row[3]))

    plt.style.use('ggplot')
    
    # plt.ylabel('y')
    # plt.title('Interesting Graph\nCheck it out')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    # fig.suptitle('Example of a Single Legend Shared Across Multiple Subplots')

    ax1.set_title('Gold Testset')
    ax2.set_title('NCBO Testset')
    ax1.set_xlabel('similarity threshold')
    ax2.set_xlabel('similarity threshold')
    # plt.xlabel('similarity score')

    l1 = ax1.plot(gsim,gf)[0]
    l2 = ax1.plot(gsim,gp)[0]
    l3 = ax1.plot(gsim,gr)[0]
    l3m = ax1.plot(gsim[6], gf[6], 'g*')[0]
    l4 = ax2.plot(nsim,nf)[0]
    l5 = ax2.plot(nsim,np)[0]
    l6 = ax2.plot(nsim,nr)[0]
    l6m = ax2.plot(nsim[0], nf[0], 'g*')[0]

    # fig.legend([l1, l2, l3,l4,l5,l6], labels=['f1', 'precision', 'recall','f1', 'precision', 'recall'])
    line_labels = ['f1', 'precision', 'recall', 'max f1']

    fig.legend([l1, l2, l3, l3m, l4, l5, l6, l6m],     # The line objects
            labels=line_labels,   # The labels for each line
            loc="center",   # Position of legend
            borderaxespad=0.1   # Small spacing around legend box
            # title="Legend Title"  # Title for the legend
            )
    plt.subplots_adjust(wspace = 0.7)

    # plt.subplot(1, 2, 1)
    # plt.plot(gsim,gf, label='F1')
    # plt.plot(gsim,gp, label='Precision')
    # plt.plot(gsim,gr, label='Recall')
    # plt.plot(gsim[6], gf[6], 'g*', label='Max F1')
    # plt.title('Gold Testset')


    # plt.subplot(1, 2, 2)
    # plt.plot(nsim,nf, label='F1')
    # plt.plot(nsim,np, label='Precision')
    # plt.plot(nsim,nr, label='Recall')
    # plt.plot(nsim[0], nf[0], 'g*')
    # plt.title('NCBO Testset')

    # plt.xlabel('similarity score')
    # plt.legend()
    plt.show()

# _draw_plots()