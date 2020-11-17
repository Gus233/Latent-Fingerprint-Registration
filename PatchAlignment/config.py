class Config(object):
    env = 'default'
    backbone = 'align'

    finetune = False

    #####

    train_root = '/media/disk3/gs/LatentMatchData/Alignment/train_patch_pair_ave_50/'
    pd_root = '/media/disk3/gs/LatentMatchData/Alignment/train_patch_pair_ave_orientation_50/'
    train_list = '/media/disk3/gs/LatentMatchData/Alignment/train_patch_pair_ave_50_select.txt'

  

    checkpoints_path = '/media/disk3/gs/LatentMatchData/Alignment/checkpoints/'
    load_model_path =  '/media/disk3/gs/LatentMatchData/Alignment/checkpoints/50/align_25.pth'


    ######

    test_root = '/media/disk3/gs/LatentMatchData/Alignment/AvePara/test_patch_pair_ave_para_ave_50/'
    test_pd_root = '/media/disk3/gs/LatentMatchData/AlignmentAvePara//test_patch_pair_ave_para_ave_orientation_50/'
    test_list = '/media/disk3/gs/LatentMatchData/Alignment/AvePara/test_patch_pair_ave_para_ave_50_select.txt'

    test_model_path = '/media/disk3/gs/LatentMatchData/Alignment/checkpoints/align_15.pth'
    save_path = '/media/disk3/gs/LatentMatchData/Alignment/test_patch_pair_ave_para_ave_50.mat'





    save_interval = 5

    train_batch_size = 32  # batch size
    test_batch_size = 20

    input_shape = (1, 200, 200)

    optimizer = 'adam'

    use_gpu = True  # use GPU or not
    gpu_id = '3'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch



    max_epoch = 31
    lr = 1e-3 # initial learning rate   ############
    lr_step = 5
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
