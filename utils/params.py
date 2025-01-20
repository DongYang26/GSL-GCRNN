import argparse
import sys


argv = sys.argv
dataset = argv[1]


def bcn_L_params():
    parser = argparse.ArgumentParser()
    #####################################
    # basic info
    parser.add_argument("--data", type=str, help="The name of the dataset", default="spectrum")
    parser.add_argument("--model_name",type=str, default="GSLGCRNN")
    parser.add_argument("--settings",type=str, default="supervised")
    parser.add_argument("--log_path", type=str, default='./log', help="Path to the output console log file")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=3000)
    # data
    parser.add_argument("--batch_size", type=int, default=1596)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--pre_len", type=int, default=1)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--normalize", type=bool, default=True)
    # tasks
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    # model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--com_lambda_v1", type=float, default=0.1)  # for view estimator
    parser.add_argument("--com_lambda_v2", type=float, default=0.1)  # for view estimator
    parser.add_argument("--ve_dropout", type=float, default=0.2)  # for view estimator
    # data path
    parser.add_argument("--feat_path", type=str, default="./Data/trainData/bcn_L/bcn-L_202106010800-202106080800_1mins_2MHz.csv")  # for view estimator
    parser.add_argument("--testDataPath", type=str, default="./Data/testData/bcn_L/bcn-L_202106010800-202106080800_1mins_2MHz.csv")  # for view estimator
    parser.add_argument("--view1Path", type=str, default="./Data/trainData/bcn_L/"+"v2_adj.npz")  # for view estimator
    parser.add_argument("--view2Path", type=str, default="./Data/trainData/bcn_L/"+"v3_diff.npz")
    parser.add_argument("--v1_indicesPath", type=str, default="./Data/trainData/bcn_L/"+"v2_2.pt")
    parser.add_argument("--v2_indicesPath", type=str, default="./Data/trainData/bcn_L/"+"v3_5.pt")

    #####################################

    temp_args, _ = parser.parse_known_args()
    return temp_args


def test_rpi4_params():
    parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    #####################################
    # basic info
    parser.add_argument("--data", type=str, help="The name of the dataset", default="spectrum")
    parser.add_argument("--model_name", type=str, default="GSLGCRNN")
    parser.add_argument("--settings", type=str, default="supervised")
    parser.add_argument("--log_path", type=str, default='./log', help="Path to the output console log file")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=3000)
    # data
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pre_len", type=int, default=1)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--normalize", type=bool, default=True)
    # tasks
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    # model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--com_lambda_v1", type=float, default=0.1)  # for view estimator
    parser.add_argument("--com_lambda_v2", type=float, default=0.1)  # for view estimator
    parser.add_argument("--ve_dropout", type=float, default=0.2)  # for view estimator
    # data path
    parser.add_argument("--feat_path", type=str,
                        default="./Data/trainData/test_rpi4/test_rpi4_202106010800-202106080800_1mins_2MHz.csv")  # for view estimator
    parser.add_argument("--testDataPath", type=str,
                        default="./Data/testData/test_rpi4/test_rpi4_202106010800-202106080800_1mins_2MHz.csv")  # for view estimator
    parser.add_argument("--view1Path", type=str, default="./Data/trainData/test_rpi4/"+"v2_adj.npz")  # for view estimator
    parser.add_argument("--view2Path", type=str, default="./Data/trainData/test_rpi4/"+"v3_diff.npz")
    parser.add_argument("--v1_indicesPath", type=str, default="./Data/trainData/test_rpi4/"+"v2_2.pt")
    parser.add_argument("--v2_indicesPath", type=str, default="./Data/trainData/test_rpi4/"+"v3_5.pt")

    #####################################

    temp_args, _ = parser.parse_known_args()
    return temp_args


def rack_2_params():
    parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    #####################################
    # basic info
    parser.add_argument("--data", type=str, help="The name of the dataset", default="spectrum")
    parser.add_argument("--model_name", type=str, default="GSLGCRNN")
    parser.add_argument("--settings", type=str, default="supervised")
    parser.add_argument("--log_path", type=str, default='./log', help="Path to the output console log file")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=3000)
    # data
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pre_len", type=int, default=1)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--normalize", type=bool, default=True)
    # tasks
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    # model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--com_lambda_v1", type=float, default=0.1)  # for view estimator
    parser.add_argument("--com_lambda_v2", type=float, default=0.1)  # for view estimator
    parser.add_argument("--ve_dropout", type=float, default=0.2)  # for view estimator
    # data path
    parser.add_argument("--feat_path", type=str,
                        default="./Data/trainData/rack_2/rack_2_202106010800-202106080800_1mins_2MHz.csv")  # for view estimator
    parser.add_argument("--testDataPath", type=str,
                        default="./Data/testData/rack_2/rack_2_202106010800-202106080800_1mins_2MHz.csv")  # for view estimator
    parser.add_argument("--view1Path", type=str, default="./Data/trainData/rack_2/"+"v2_adj.npz")  # for view estimator
    parser.add_argument("--view2Path", type=str, default="./Data/trainData/rack_2/"+"v3_diff.npz")
    parser.add_argument("--v1_indicesPath", type=str, default="./Data/trainData/rack_2/"+"v2_2.pt")
    parser.add_argument("--v2_indicesPath", type=str, default="./Data/trainData/rack_2/"+"v3_5.pt")

    #####################################

    temp_args, _ = parser.parse_known_args()
    return temp_args


def test_yago_params():
    parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    #####################################
    # basic info
    parser.add_argument("--data", type=str, help="The name of the dataset", default="spectrum")
    parser.add_argument("--model_name", type=str, default="GSLGCRNN")
    parser.add_argument("--settings", type=str, default="supervised")
    parser.add_argument("--log_path", type=str, default='./log', help="Path to the output console log file")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=3000)
    # data
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pre_len", type=int, default=1)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--normalize", type=bool, default=True)
    # tasks
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    # model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--com_lambda_v1", type=float, default=0.1)  # for view estimator
    parser.add_argument("--com_lambda_v2", type=float, default=0.1)  # for view estimator
    parser.add_argument("--ve_dropout", type=float, default=0.2)  # for view estimator
    # data path
    parser.add_argument("--feat_path", type=str,
                        default="./Data/trainData/test_yago/test_yago_202106010800-202106080800_1mins_2MHz.csv")  # for view estimator
    parser.add_argument("--testDataPath", type=str,
                        default="./Data/testData/test_yago/test_yago_202106010800-202106080800_1mins_2MHz.csv")  # for view estimator
    parser.add_argument("--view1Path", type=str, default="./Data/trainData/test_yago/"+"v2_adj.npz")  # for view estimator
    parser.add_argument("--view2Path", type=str, default="./Data/trainData/test_yago/"+"v3_diff.npz")
    parser.add_argument("--v1_indicesPath", type=str, default="./Data/trainData/test_yago/"+"v2_2.pt")
    parser.add_argument("--v2_indicesPath", type=str, default="./Data/trainData/test_yago/"+"v3_5.pt")

    #####################################

    temp_args, _ = parser.parse_known_args()
    return temp_args


def set_params():
    if dataset == "bcn_L":
        args = bcn_L_params()
    elif dataset == "test_rpi4":
        args = test_rpi4_params()
    elif dataset == "rack_2":
        args = rack_2_params()
    elif dataset == "test_yago":
        args = test_yago_params()
    else:
        raise ValueError(f"The dataset '{dataset}' does not exist.")
    return args
