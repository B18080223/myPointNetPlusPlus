"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40], help='training on ModelNet10/40')
    # parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()

# def test(model, loader, num_class=40, vote_num=1):
def test(model, loader, num_class, vote_num=1):
    i = 0
    result=[]
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1) #转置
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]
        # result.append(pred_choice)
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # print(result)
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    # 可视化
    file_dir = 'E:\Desktop\temp'
    save_name_prefix = 'pred'
    n_points=points.cpu().numpy()
    # draw(n_points[:, 0, :], n_points[:, 1, :], n_points[:, 2, :], save_name_prefix, file_dir, color=pred_choice)

    return instance_acc, class_acc




#定义可视化函数
def draw(x, y, z, name, file_dir, color=None):
    """
    绘制单个样本的三维点图
    """
    if color is None:
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
            save_name = name + '-{}.png'.format(i)
            save_name = os.path.join(file_dir,save_name)
            ax.scatter(x[i], y[i], z[i],s=0.1, c='r')
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw()
            plt.savefig(save_name)
            # plt.show()
    else:
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'tan', 'orangered', 'lightgreen', 'coral', 'aqua']
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
            save_name = name + '-{}-{}.png'.format(i, color[i])
            save_name = os.path.join(file_dir,save_name)
            ax.scatter(x[i], y[i], z[i],s=0.1, c=colors[color[i]])
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw()
            plt.savefig(save_name)
            # plt.show()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER 超参数'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)  #存储评估结果
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
