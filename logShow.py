import matplotlib.pyplot as plt
import csv
import numpy as np
import glob

def plotData(allData, legend, title, loc="lower right"):
    #print(data)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:purple']

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.subplots_adjust(hspace=0.15)
    #fig.suptitle('Left Title', horizontalalignment='left')
    #fig.suptitle('Right Title', horizontalalignment='right')
    print(len(allData))
    for mode, data in enumerate(allData):
        print("OUT OF MODE")
        for k, d in enumerate(data):
            x, avg = d
            print(colors[mode], mode)
            print(x)
            axs[k].plot(20*k+np.arange(x.shape[0]) ,x, color=colors[mode])
            axs[k].fill_between(20*k+np.arange(x.shape[0]),x-avg, x+avg, alpha=0.5, color=colors[mode])
            #axs[k].grid()
            axs[k].get_xaxis().set_ticks([])
            axs[k].set_ylabel('Task '+str(k+1))
            axs[k].spines['bottom'].set_visible(False)
            axs[k].spines['top'].set_visible(False)
            axs[k].set_ylim(80, 100)
            axs[k].set_xlim(0, 60)
            axs[k].axvline(x = 20,linewidth=1, color='black', linestyle='--')
            axs[k].axvline(x = 40,linewidth=1, color='black', linestyle='--')
            if k == 0:
                axs[k].set_title('          Train A', loc='left')
                axs[k].set_title('Train B', loc='center')
                axs[k].set_title('Train C           ', loc='right')
        leg = plt.legend(legend,bbox_to_anchor=(1.01, 3), loc=2, borderaxespad=0, handlelength=0, handletextpad=0, fancybox=True)
        #Hide Legend linestyle
        for item in leg.legendHandles:
            item.set_visible(False)
        #Change colors
        for cont, text in enumerate(leg.get_texts()):
            text.set_color(colors[cont])

        #plt.title(title)
    plt.savefig(title+'.pdf')
    plt.show()

def openFile(file):
    with open(file, 'r') as f:
        readCSV = csv.reader(f, delimiter=',')
        return readCSV

def getData(path):
    print(path, "PATH")
    files  = glob.glob(path+'/*')
    D = {'test_acc':[]}

    print(files)
    for f in files:
        print(f)
        with open(f, 'r') as f:
            readCSV = csv.reader(f, delimiter=',')
            test_acc = list()
            for line in readCSV:
                print(line)
                #line.split(',')
                #train_acc.append(float(line[2]))
                test_acc.append(float(line[5]))
            D['test_acc'].append(np.array(test_acc))
    test_acc = np.array(D['test_acc'])


    D['test_acc'] = [np.average(test_acc, axis=0), np.std(test_acc, axis=0)]


    return D


def main():
    files = "not_sure"
    dataTest = list()
    dataTrain = list()
    dataDistLoss = list()
    #
    # files = ["None_1e-4_D2", "ED_1e-4_D2", "ED_1e-4_D13", "ED_1e-8_D13", "ED_1e-4_D-3_cifar10", "ED_1e-4_D-3_TO_10", "ED_1e-4_D-3_TO_25", ]
    # legends = ["Normal", "Euclidean_1e-4_2", "Euclidean_1e-4_13", "Euclidean_1e-8_13", "ED_1e-4_D-3", "ED_1e-4_D-3_TO_10", "ED_1e-4_D-3_TO_25"]
    # for f in files:
    #     data = getData('./log/'+f)
    #     #print(data)
    #     dataTrain.append(np.array(data["train_acc"]))
    #     dataTest.append(np.array(data["test_acc"]))
    #
    # plotData(dataTrain, legends, 'Train')
    # plotData(dataTest, legends, 'Test')

    # files = ["None_1e-4_D13_cifar100", "ED_1e-4_D-3_TO_10_cifar100", "ED_1e-4_D-1_TO_10_cifar100_resnet18", "None_1e-4_D20_TO_None_cifar100_resnet", "resnet_ED_D20_1e-8_TO25_outd0.1_anneal_true_cifar100", "resnet_none_DNone__cifar_100", "resnet18_ED_1e-07_D-1_TO_10_cifar100","resnet_ED_1e-4_D20_TO_25_outd_0.1_anneal_true_cifar100"]
    # legends = ["None_1e-4_D13_cifar100", "ED_1e-4_D-3_TO_10_cifar100", "ED_1e-4_D-1_TO_10_cifar100_resnet18", "None_1e-4_D20_TO_None_cifar100_resnet<-NORMAL TRAIN", "resnet_ED_D20_1e-8_TO25_outd0.1_anneal_true_cifar100", "resnet_none_DNone__cifar_100<-NORMAL TRAIN ANNEAL LR", "resnet18_ED_1e-07_D-1_TO_10_cifar100", "resnet_ED_1e-4_D20_TO_25_outd_0.1_anneal_true_cifar100"]
    # for f in files:
    #     data = getData('./log/cifar100/'+f)
    #     #print(data)
    #     dataTrain.append(np.array(data["train_acc"]))
    #     dataTest.append(np.array(data["test_acc"]))
    #
    # plotData(dataTrain, legends, 'Train')
    # plotData(dataTest, legends, 'Test')

    allFiles = [["Dropout_MNIST_TASK_1", "Dropout_MNIST_TASK_2", "Dropout_MNIST_TASK_3"], ["SGD_MNIST_TASK_1", "SGD_MNIST_TASK_2", "SGD_MNIST_TASK_3"]]
    legends = ["Dropout", "SGD"]

    allData = []

    for files in allFiles:
        dataTest = list()
        for f in files:
            data = getData('./log/'+f)
            #print(data)
            dataTest.append(np.array(data["test_acc"]))

        allData.append(dataTest)

    #plotData(dataTrain, legends, 'Train')
    plotData(allData, legends, 'Test')

    # files = ["off", "on" ]
    # legends = ["AlexNet_ED_OFF", "AlexNet_ED_ON"]
    # for f in files:
    #     data = getData('./log/EUCLIDEAN_OFF_ON/'+f)
    #     #print(data)
    #     dataDistLoss.append(data['dist_loss'])
    #
    # plotData(dataDistLoss, legends, 'Euclidean Distance', loc='upper left')

if __name__ == "__main__":
    main()
