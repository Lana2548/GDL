from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pickle

data_dir = "weights/"

def my_eval(model, x_test, y_test):
    ev = model.evaluate(x_test, y_test)
    print("loss:", end = " ")
    print(ev[0])
    print("acc:", end = "")
    print(ev[1])
## historyの可視化．nameを指定した場合は historiesに引数として与えたhistory， loss_and_accに可視化結果をそれぞれ保存
def loss_and_acc(history,file_name = None):
    fig,ax = plt.subplots(1,2,figsize = (10,5))
    epochs = len(history.history["loss"])
    ax[0].plot(range(epochs), history.history["loss"], label = "train_loss", c = "tomato")
    ax[0].plot(range(epochs), history.history["val_loss"], label = "valid_loss", c = "c")
    ax[0].set_xlabel("epochs",fontsize = 14)
    ax[0].set_ylabel("loss",fontsize  = 14)
    ax[0].legend(fontsize = 14)

    ax[1].plot(range(epochs), history.history["accuracy"], label="train_acc", c="tomato")
    ax[1].plot(range(epochs), history.history["val_accuracy"], label="valid_acc", c="c")
    ax[1].set_xlabel("epochs", fontsize=14)
    ax[1].set_ylabel("acc", fontsize=14)
    ax[1].legend(fontsize = 14)
    """
    if(file_name != None):
        with open(data_dir + "histories/" + file_name + ".binaryfile",mode = "wb") as f:
            pickle.dump(history, f)
    if(file_name != None):
        fig.savefig(data_dir + "loss_and_acc/" + file_name + "_acc" )
    """


## historyのロード  loss_and_acc()で与えたfile_nameを指定すると，そのhistoryをロードして返り値とする
def load_history(file_name):
    with open(data_dir + "histories/" + file_name + ".binaryfile",mode = "rb") as f:
        res = pickle.load(f)
    return res


## 2つのhistoryの比較と保存
def compare(his1,his1_name,his2,his2_name, file_name = None):
    #his1: 比較したいヒストリー , his1_name: his1のラベル   his2も同様
    #file_name  与えると，可視化結果を保存
    keys = ["loss","val_loss","acc","val_acc"]
    fig, ax = plt.subplots(2,2,figsize = (12,12))
    epochs = min( [len(his1.history["loss"]), len(his2.history["loss"])] )

    ind = 0
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(range(epochs),his1.history[keys[ind]][:epochs],label = his1_name)
            ax[i,j].plot(range(epochs),his2.history[keys[ind]][:epochs],label = his2_name)
            ax[i,j].set_xlabel("epochs",fontsize = 14)
            ax[i,j].set_ylabel(keys[ind],fontsize = 14)
            ax[i,j].legend(fontsize = 14)

            ind += 1

    if(file_name != None):
        fig.savefig(data_dir + "comparisons/" + file_name + "_comp")