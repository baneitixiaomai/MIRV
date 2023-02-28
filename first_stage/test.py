import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from scipy import misc
from eval_func import *
from data import test_dataset_rgbd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from model.ResNet_VGG_models import Generator2


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')

opt = parser.parse_args()

            
def test(result_txt,generator_path,result_root,testsize=352):
    dataset_path = './test/'
    pre_root_rgbd = result_root + '_rgbd_test' + '/'

    model = Generator2(channel=32)
    model.load_state_dict(torch.load(generator_path))

    model.cuda()
    model.eval()

    test_datasets =['NJU2K','STERE','DES','NLPR','LFSD','SIP']

    for dataset in test_datasets:

        save_path_rgbd = pre_root_rgbd + dataset + '/'
        if not os.path.exists(save_path_rgbd):
            os.makedirs(save_path_rgbd)

        image_root = dataset_path + dataset + '/RGB/'
        depth_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset_rgbd(image_root,depth_root, testsize)
        for i in tqdm(range(test_loader.size)):
            image, depth, HH, WW, name = test_loader.load_data()
            image = image.cuda()
            depth = depth.cuda()
            sal_rgbd= model(image,depth)[1]
            res = F.upsample(sal_rgbd, size=[WW,HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path_rgbd+name, res)
            pass


    result_txt_rgbd = result_txt +'_rgbd.txt'
    f = open(result_txt_rgbd, 'w+')
    gt_dir=dataset_path
    # test_datasets = ["NC4K"]
    print('[INFO]: Process in Path [{}]'.format(pre_root_rgbd) , file=f)

    results_list = []
    mm_list = []
    columns_pd = ['S_measure', 'F_measure', 'E_measure', 'MAE'] 

    for dataset in test_datasets:
        print("[INFO]: Process {} dataset".format(dataset) , file=f)
        if dataset != 'SOD':
            loader = eval_Dataset(osp.join(pre_root_rgbd, dataset), osp.join(gt_dir, dataset, 'GT'))
        else:
            loader = eval_SODDataset(osp.join(pre_root_rgbd, dataset), osp.join(gt_dir, dataset, 'GT'))

        def my_collate(batch):
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            return [data, target]
        data_loader = data.DataLoader(dataset=loader, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, collate_fn=my_collate)
        
        torch.cuda.synchronize(); start = time()
        [mae, F_measure, S_measure, E_measure] = eval_batch(loader=data_loader)
        torch.cuda.synchronize(); end = time()

        # print('[INFO] Time used: {:.4f}'.format(end - start))
        measure_list = np.array([S_measure, F_measure.item(), E_measure.item(), mae.item()])
        print(pd.DataFrame(data=np.reshape(measure_list, [1, len(measure_list)]), columns=columns_pd).to_string(index=False, float_format="%.5f"), file=f)
        results_list.append(measure_list)
        for kk in range(4):
            # num = '.'+str(np.around(measure_list[kk], 3)).split(('.'))[-1]
            mm_list.append(str(measure_list[kk]))
 
    
    result_table = pd.DataFrame(data=np.vstack((results_list)), columns=columns_pd, index=test_datasets)
    print(result_table.to_string(float_format="%.3f"), file=f)
    f.close()

model_save_path = './models/'
result_txt = model_save_path+'result'
generator_path = model_save_path + 'scribble_30'  + '.pth'
test(result_txt,generator_path,result_root= model_save_path+'/results',testsize=352)
