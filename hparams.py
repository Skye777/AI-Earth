import argparse


class Hparams:
    parser = argparse.ArgumentParser()
    # test
    parser.add_argument('--path_list', default=['./tcdata/enso_round1_test_20210201/',
                                                '/home/dl/Public/Skye/AI-Earth/test样例_20210207_update/test样例'])
    parser.add_argument('--delivery_model_dir', default='ckpt/checkpoints')
    parser.add_argument('--delivery_model_file', default='ckp_1')

    # prepro
    parser.add_argument('--npz_dir', default='/home/dl/Public/Skye/AI-Earth/final_data')
    parser.add_argument('--preprocess_out_dir',
                        default='/home/dl/Public/Skye/AI-Earth/final_data/tfRecords')

    # data
    parser.add_argument('--in_seqlen', default=12)
    parser.add_argument('--out_seqlen', default=24)
    parser.add_argument('--lead_time', default=1)
    parser.add_argument('--width', default=72)
    parser.add_argument('--height', default=24)
    parser.add_argument('--num_predictor', default=4)
    parser.add_argument('--input_variables', default=["sst", "t300", "ua", "va"])
    parser.add_argument('--output_variables', default=["sst", "t300", "ua", "va"])

    # training scheme
    parser.add_argument('--train_eval_split', default=0.2)
    parser.add_argument('--random_seed', default=2021)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--early_stop_patience', default=20)
    parser.add_argument('--num_epoch_record', default=1, help="Number of step to record checkpoint.")

    parser.add_argument('--ckpt', default='', help="checkpoint file path")
    parser.add_argument('--multi_gpu_model_dir', default="ckpt/checkpoints")
    parser.add_argument('--single_gpu_model_dir', default="ckpt/checkpoints")
    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="logs", help="log directory")

    # model
    parser.add_argument('--model_structure', default="Joint")
    parser.add_argument('--vunits', default=108)
    parser.add_argument('--Tunits', default=8)
    parser.add_argument('--Munits', default=8)
    parser.add_argument('--MTunits', default=8)
    parser.add_argument('--V_kernel', default=3)
    parser.add_argument('--V_stride', default=1)
    parser.add_argument('--d_model', default=864, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=100, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=100, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")
