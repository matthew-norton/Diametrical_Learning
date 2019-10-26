import matplotlib.pyplot as plt
import numpy as np



def save_nets(fname,network1,network2,optimizer1,optimizer2,lr_schedule,g_schedule):
    network1_init = { 'v' : list(network1.v) ,
                     'train_loss' : network1.train_loss ,
                     'test_loss' : network1.test_loss ,
                     'test_acc' : network1.test_acc ,
                     'train_steps' : network1.train_steps ,
                     'model_params' : network1.state_dict(),
                     'optimizer_params':optimizer1.state_dict() ,

                     }

    network2_init = { 'v' : list(network2.v) ,
                     'train_loss' : network2.train_loss ,
                     'test_loss' : network2.test_loss ,
                     'test_acc' : network2.test_acc ,
                     'train_steps' : network2.train_steps ,
                     'model_params':network2.state_dict() ,
                     'optimizer_params':optimizer2.state_dict()

                    }
    all_info = { 'DRM': network1_init ,
                  'ERM':network2_init,
                  'percent_flip':percent_flip,
                  'v_max_len' : network1.v_max_len,
                  'num_dir':network1.num_dir,
                  'batch_size_train':batch_size_train ,
                  'lr_schedule' : lr_schedule ,
                 'g_schedule':g_schedule

               }



    torch.save(all_info, fname)


#############################################################################################
'''PLOT LOSS AND ACCURACY'''
#############################################################################################



def plot_net(network_list,fname = None):
    '''Expects a list of tuples [(network,name_of_network) , ..., ] where name_of_network
    is used for the plot legend'''
    assert len(network_list)<=4 , "not enough colors in variable 'colors' for more than 4 plots."

    fig =plt.figure(dpi=100)
    fig.set_size_inches(4.5*2, 2.25*2)

    colors = ['r','b','g','y']
    for i,(network,name) in enumerate(network_list):
        style = '-{}'.format(colors[i])
        plt.plot(np.array(network.test_acc)[1:,1] , np.array(network.test_acc)[1:,0] , style ,label = name  )
    plt.legend()
    plt.title('Test Acc. During Training' )#(800 epochs) \n 3-class CIFAR ')
    plt.xlabel('Train Steps')
    plt.ylabel('Test Acc. (%)')
    if fname is not None:
        plt.savefig(fname, dpi=600)
    else:
        plt.savefig('temp_plot.pdf',dpi=600)
    plt.close()

'''
all_info = torch.load('/home/matthewnorton/Documents/torch-tests/Diametrical_Learning/cifar_res/CIFAR_simple.pkl.pkl')

network1 = Net(3, (3,32,32) , num_dir = 20 ).to(device)
model_info=all_info['DRM']
network1.load_state_dict(model_info['model_params'])
network1.train_loss=model_info['train_loss']
network1.test_loss=model_info['test_loss']
network1.test_acc=model_info['test_acc']
network1.train_steps=model_info['train_steps']
network1.v = deque(model_info['v'] , maxlen=len(model_info['v']) )

network2 = Net(3, (3,32,32) ).to(device)
model_info=all_info['ERM']
network2.load_state_dict(model_info['model_params'])
network2.train_loss=model_info['train_loss']
network2.test_loss=model_info['test_loss']
network2.test_acc=model_info['test_acc']
network2.train_steps=model_info['train_steps']
network2.v = deque(model_info['v'] , maxlen=len(model_info['v']) )
'''
