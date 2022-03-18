import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import os
from Network import AttentiontTransformer
import warnings
import utils
from scipy.io import savemat
import logging
import yaml

warnings.filterwarnings("ignore")


def main(args):
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    print()

    with open(args.config) as file:
        yaml_config = yaml.safe_load(file)
    data_path = yaml_config['data_path']
    label_path = yaml_config['label_path']
    result_save_path = yaml_config['result_save_path']
    save_path = yaml_config['save_path']
    subject = yaml_config['subject']

    train_data, test_data = utils.loaddata(data_path, label_path)
    # list
    train_lossesa, val_losses = list(), list()
    pred_tests, labels = list(), list()

    lossweight = torch.tensor([args.lw_0, args.lw_1, args.lw_2], dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=lossweight).to(device)

    model = AttentiontTransformer(eeg_channel=args.EEG_Channel, d_model=args.d_model, n_head=args.n_head,
                                  d_hid=args.d_hid, n_layers=args.n_layers, dropout=args.dropout, max_len=799,
                                  device=device).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0)

    logfilename = subject + '_loss.log'

    logging.basicConfig(filename=logfilename, level=logging.INFO)
    logging.info('start')

    with trange(args.epoch_num, unit='iteration', desc='epoch') as pbar:
        for epoch in range(args.epoch_num):
            train_losses = list()
            for i, content in enumerate(train_data):
                # Prepare Data
                data = torch.tensor(content['eeg'], dtype=torch.float32).to(device)
                label = torch.tensor(content['label'], dtype=torch.long).to(device)
                # Forward Data #
                pred = model(data)
                if label.shape[0] == 1:
                    label = label.squeeze(0)
                else:
                    label = label.squeeze()
                # Calculate Loss #
                train_loss = criterion(pred, label)
                # Initialize Optimizer, Back Propagation and Update #
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                # Add item to Lists #
                train_losses.append(train_loss.item())
            loss = np.average(train_losses)
            pbar.set_postfix(loss=loss)
            pbar.n += 1
            train_lossesa.append(loss)
            logging.info(loss)

            # Learning Rate Scheduler #
            optim_scheduler.step()

            if epoch % args.savenum == 0:
                torch.save(model.state_dict(),
                           os.path.join(save_path, (subject + '_Model_Using_Attention_' + str(epoch) + '.pkl')))

    # plt.plot(train_lossesa)
    # plt.show()
    torch.save(model.state_dict(),
               os.path.join(save_path, (subject + '_Final_Model_Using_Attention_' + str(args.epoch_num) + '.pkl')))

    model.load_state_dict(
        torch.load(os.path.join(save_path, (subject + '_Final_Model_Using_Attention_' + str(args.epoch_num) + '.pkl'))))

    with torch.no_grad():
        for i, content in enumerate(test_data):
            # Prepare Data #
            data = torch.tensor(content['eeg'], dtype=torch.float32).to(device)
            label = torch.tensor(content['label'], dtype=torch.long).to(device)
            # Forward Data #
            pred_test = model(data)
            pred_tests += pred_test.tolist()
            labels += label.tolist()

        # plt.plot(pred_tests)
        # plt.plot(labels)
        # plt.legend(['prict', 'label'])
        # plt.show()

        # # Save mat files #
        pred_tests = np.asarray(pred_tests)
        pred_tests = {'pred_test': pred_tests}
        labels = np.asarray(labels)
        labels = {'label': labels}
        pred_mat_filename = result_save_path + '/' + subject + '_pred_tests.mat'
        label_mat_filename = result_save_path + '/' + subject + '_label.mat'
        savemat(pred_mat_filename, pred_tests)
        savemat(label_mat_filename, labels)
        logging.info('end')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--EEG_Channel', type=float, default=17, help='the number of channel of EEG dataset')

    parser.add_argument('--n_head', type=float, default=8, help='number of head used in multi_head attention algorithm')
    parser.add_argument('--d_model', type=float, default=32,
                        help='the dimension of data put in to multi_head attention algorithm')
    parser.add_argument('--d_hid', type=float, default=2048, help='the dimension of feedforward network model')
    parser.add_argument('--n_layers', type=float, default=3, help='the number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout value')

    parser.add_argument('--epoch_num', type=int, default=10, help='total epoch')
    parser.add_argument('--savenum', type=int, default=5, help='total epoch')
    parser.add_argument('--config', type=str,
                        default='/Users/mike/PycharmProjects/pythonProjectsshtest/attention_test_arc/Configs/10_20151125_noon.yaml',
                        help='config_path')
    parser.add_argument('--lw_0', type=int, default=1, help='loss weight of class0')
    parser.add_argument('--lw_1', type=int, default=1, help='loss weight of class1')
    parser.add_argument('--lw_2', type=int, default=1, help='loss weight of class2')

    config = parser.parse_args()

    main(config)
