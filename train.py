import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer
from net import *


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/train_val_test_data/pickup', help='data path')

parser.add_argument('--adj_data', type=str, default='data/sensor_graph/all-graph-distance.pkl', help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,
                    help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False, help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')

parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=406, help='number of nodes/variables')
parser.add_argument('--bike_nodes', type=int, default=221, help='number of bike nodes')
parser.add_argument('--taxi_nodes', type=int, default=185, help='number of taxi nodes')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=50, help='k')
parser.add_argument('--node_dim', type=int, default=16, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')

parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
parser.add_argument('--end_channels', type=int, default=128, help='end channels')

parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=48, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=3, help='output sequence length')

parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
parser.add_argument('--step_size2', type=int, default=150, help='step_size')

parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--save', type=str, default='./save/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--propalpha', type=float, default=0.0, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')

parser.add_argument('--runs', type=int, default=3, help='number of runs')

args = parser.parse_args()
torch.set_num_threads(3)


def main(runid):
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, args.bike_nodes, args.taxi_nodes)
    scaler_bike = dataloader['scaler_bike']
    scaler_taxi = dataloader['scaler_taxi']

    predefined_A = load_adj(args.adj_data)
    predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    if args.load_static_feature:
        static_feat = load_node_feature('data/sensor_graph/location.csv')
    else:
        static_feat = None
    l_matrix = [args.bike_nodes, args.taxi_nodes]
    bike_adj = load_adj('data/sensor_graph/bike-graph.pkl')
    bike_adj = torch.tensor(bike_adj).to(device)
    taxi_adj = load_adj('data/sensor_graph/taxi-graph.pkl')
    taxi_adj = torch.tensor(taxi_adj).to(device)
    pre_adj_list = [bike_adj, taxi_adj]

    model = CoGNN(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes, l_matrix, pre_adj_list,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len,
                     args.bike_nodes, args.taxi_nodes, scaler_bike, scaler_taxi, device, args.cl)   # 见trainer.py

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            # 这里有步打乱节点顺序的操作,注意区分bike和taxi
            if iter % args.step_size2 == 0:
                perm_bike = np.random.permutation(range(args.bike_nodes))
                perm_taxi = np.random.permutation(range(args.bike_nodes, args.bike_nodes+args.taxi_nodes))
                perm = np.concatenate([perm_bike, perm_taxi])
                # perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes / args.num_split)
            for j in range(args.num_split):
                if j != args.num_split - 1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                # id = torch.tensor(id).to(device)
                id = torch.tensor(id).type(torch.int64).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                metrics = engine.train(tx, ty[:, 0, :, :], id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)

        if mvalid_loss < minl:
            torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth")
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth"))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    # 在这里反标准化
    pred_bike = scaler_bike.inverse_transform(yhat[:, :args.bike_nodes, :])
    pred_taxi = scaler_taxi.inverse_transform(yhat[:, -args.taxi_nodes:, :])
    pred = torch.cat([pred_bike, pred_taxi], 1)
    vmae, vmape, vrmse = metric(pred, realy)

    # test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        # 在这里反标准化2
        pred_bike = scaler_bike.inverse_transform(yhat[:, :args.bike_nodes, i])
        pred_taxi = scaler_taxi.inverse_transform(yhat[:, -args.taxi_nodes:, i])
        pred = torch.cat([pred_bike, pred_taxi], 1)
        real = realy[:, :, i]       # shape:(862,406)

        pred_bike = pred[:, :args.bike_nodes]
        pred_taxi = pred[:, -args.taxi_nodes:]
        real_bike = real[:, :args.bike_nodes]
        real_taxi = real[:, -args.taxi_nodes:]

        metrics_bike = metric(pred_bike, real_bike)
        metrics_taxi = metric(pred_taxi, real_taxi)
        metrics = metric(pred, real)

        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        log = 'Evaluate best model on test bike-data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics_bike[0], metrics_bike[1], metrics_bike[2]))
        log = 'Evaluate best model on test taxi-data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics_taxi[0], metrics_taxi[1], metrics_taxi[2]))

        #pred_save = pred.cpu().numpy()
        #real_save = real.cpu().numpy()
        #np.save(args.save+'pred_exp'+ str(args.expid) + "_" + str(runid) +"step"+str(i+1)+ ".npy", pred_save)
        #np.save(args.save+'real_exp'+ str(args.expid) + "_" + str(runid) +"step"+str(i+1)+ ".npy", real_save)

        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    return vmae, vmape, vrmse, mae, mape, rmse


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for i in range(args.runs):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)

    print('\n\nResults for 10 runs\n\n')
    # valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    print('\n\n')
    # test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [0, 1, 2]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i + 1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))
