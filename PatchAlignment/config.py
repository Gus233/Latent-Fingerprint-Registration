class Config(object):
    env = 'default'
    backbone = 'align'

    finetune = False


    train_root = '/home/data2/gus/LatentMatch/global_train_image/'
    object_root = '/home/data2/gus/LatentMatch/global_train_image_orientation/'
    train_list = '/home/data2/gus/LatentMatch/global_train_pair.txt'

    checkpoints_path = '/home/data2/gus/LatentMatch/checkpoints/global'
 #   load_model_path = '/home/data2/gus/LatentMatch/checkpoints/global/'



    lfw_root = '/home/data2/gus/LatentMatch/nist27/test_patch_pair_NIST27_2/'
    lfw_test_list = '/home/data2/gus/LatentMatch/nist27/test_pair_NIST27_compact_ex.txt'

    test_model_path = '/home/data2/gus/LatentMatch/checkpoints/resnet18_30.pth'
    save_path = '/home/data2/gus/LatentMatch/nist27/test_pair_NIST27_2_ex.mat'





    save_interval = 5

    train_batch_size = 32  # batch size
    test_batch_size = 20

    input_shape = (1, 160, 160)

    optimizer = 'adam'

    use_gpu = True  # use GPU or not
    gpu_id = '0,1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch



    max_epoch = 101
    lr = 1e-2 # initial learning rate   ############
    lr_step = 5
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
