import  os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from eval_func import *
from data import test_dataset_rgbd
from model.ResNet_models import Pred

def test_sod(model_root,test_data_root):

    generator = Pred(channel=32, latent_dim=32)

    generator.load_state_dict(torch.load( model_root+'Model_50_gen.pth'))

    generator.cuda()
    generator.eval() 
    test_datasets = ['NJU2K','STERE','DES','NLPR','LFSD','SIP']

    for dataset in test_datasets:

        save_path_priorxd = model_root+ '/results_priorxd/' + dataset + '/'
        if not os.path.exists(save_path_priorxd):
            os.makedirs(save_path_priorxd)

        image_root = test_data_root + dataset + '/RGB/'
        depth_root = test_data_root + dataset + '/depth/'
        test_loader = test_dataset_rgbd(image_root,depth_root, 352)
        for i in tqdm(range(test_loader.size), desc=dataset):

            image, depth, HH, WW, name = test_loader.load_data()
            image = image.cuda()
            depth = depth.cuda()

            pred_priorxd = generator.forward(image,depth,training=False)

            res = torch.sigmoid(pred_priorxd)
            res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path_priorxd+name+'.png', res)

    result_txt_rgbd = model_root + 'results_priorxd_50'+'.txt'
    mae_all(result_txt_rgbd,test_data_root,model_root,test_datasets,save_film_name = 'results_priorxd')



test_data_root = './test/'
model_save_path = './models/'
result_txt = model_save_path+'result'
test_sod(model_save_path,test_data_root)