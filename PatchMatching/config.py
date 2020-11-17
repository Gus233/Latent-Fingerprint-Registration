class Config(object):
    env = 'default'
    backbone = 'descriptor'
    # number of classes of minutiae patches
    num_classes =  4530
      
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False
    contrastive = True

    ####  train
    pd_root = '/media/disk3/gs/LatentMatchData/Descriptor/minutiae_patch_orientation_select/'
    train_root = '/media/disk3/gs/LatentMatchData/Descriptor/minutiae_patch_create_origin_select/'
    train_list = '/media/disk3/gs/LatentMatchData/Descriptor/minutiae_patch_menu_select.txt'

 


    checkpoints_path = '/media/disk3/gs/LatentMatchData/Descriptor/checkpoints/' # save path
    load_model_path = '/media/disk3/gs/LatentMatchData/Descriptor/checkpoints/descriptor_30.pth'

    #####  test using nist27
    test_root = '/media/disk3/gs/LatentMatchData/Descriptor/Minutiae_Test/minutiae_test_nist27/'
    test_list = '/media/disk3/gs/LatentMatchData/Descriptor/Minutiae_Test/minutiae_test_patch_NIST27.txt'
    test_model_path = '/media/disk3/gs/LatentMatchData/Descriptor/checkpoints//descriptor_30.pth'
    save_path = '/media/disk3/gs/LatentMatchData/Descriptor/Minutiae_Test/minutiae_test_patch_nist27_l2norm.mat'


    ####
    save_interval = 5

    train_batch_size = 32  # batch size
    test_batch_size = 32 

    input_shape = (200, 200)

    optimizer = 'adam'

    use_gpu = True  # use GPU or not
    gpu_id = '1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    max_epoch = 31
    lr = 1e-2          # initial learning rate   ############
    lr_step = 5
    lr_decay = 0.95     # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
