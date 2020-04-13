class Config(object):
    env = 'default'
    backbone = 'model'
    num_classes = 4822 # number of classes of minutiae patches

    
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False
    contrastive = True

    pd_root = '/media/disk3/gs/MinutiaeDescriptor/texture_patch_orientation4/'
    train_root = '/media/disk3/gs/MinutiaeDescriptor/texture_patch_origin4/'#
    train_list = '/media/disk3/gs/MinutiaeDescriptor/texture_pair_menu4_image_select.txt'
    minu_root = '/media/disk3/gs/MinutiaeDescriptor/joint/patch_minu_map/'

    checkpoints_path = './MinutiaeDescriptor/checkpoints/' # save path

    load_model_path = './MinutiaeDescriptor/checkpoints/model_25.pth'

    test_root = '/media/disk3/gs/MinutiaeDescriptor/Arcface -contra -joint/'
    test_list = '/media/disk3/gs/MinutiaeDescriptor/Arcface -contra -joint/temp.txt'

    test_model_path = '/media/disk3/gs/MinutiaeDescriptor/checkpoints/finetune//resnet18_20.pth'  # select2
    save_path = '/home/data2/gus/LatentMatch/tmp.mat'


    save_interval = 5

    train_batch_size = 32  # batch size
    test_batch_size = 32 

    input_shape = (1, 160, 160)

    optimizer = 'adam'

    use_gpu = True  # use GPU or not
    gpu_id = '3'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    max_epoch = 41
    lr = 1e-2           # initial learning rate   ############
    lr_step = 5
    lr_decay = 0.95     # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
