

class CONFIG(object):
    def __init__(self):
        super(CONFIG, self).__init__()

        self.phase = 'train'

        self.seed = 2019
        self.device = [0] 

        self.max_epoch = 20
        self.batch_size = 56
        self.test_batch_size = 1

        # Dataset setting
        self.num_worker = 8
        self.test_csv_path = "./exp_data/TACoS/test_clip-sentvec.pkl"
        self.train_csv_path = "./exp_data/TACoS/train_clip-sentvec.pkl"
        self.test_feature_dir="../TACoS/Interval128_256_overlap0.8_c3d_fc6/"
        self.train_feature_dir = "../TACoS/Interval64_128_256_512_overlap0.8_c3d_fc6/"

        self.movie_length_info_path = "./video_allframes_info.pkl"
        
        self.context_num = 1
        self.context_size = 128

        # Model setting
        self.visual_dim = 4096 * 3
        self.sentence_embed_dim = 4800
        self.semantic_dim = 1024    # the size of visual and semantic comparison size
        self.middle_layer_dim = 1024

        self.IoU = 0.5
        self.nIoU = 0.15

        # Optimizer settking
        self.optimizer = 'Adam'
        self.vs_lr = 5e-3
        self.weight_decay = 1e-5

        self.lambda_reg = 0.01
        self.alpha = 1.0 / self.batch_size

    
        # Logging setting
        self.save_log = False
        self.log_path = './log.txt'

        self.test_output_path = "./ctrl_test_results.txt"




