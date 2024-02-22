#ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from numpy import roll
from torch.nn.functional import conv1d


def roll(data,shift):
    assert data.shape[0] == shift.shape[0]
    shift = shift-shift.min()
    re = np.zeros_like(data)
    for i in range(data.shape[0]):
        re[i] = np.roll(data[i],shift[i],axis=0)
    return re
# 给出标准的tmf函数

def cc(data,tmp,step=1):
    tmf_num = tmp.shape[0]
    groups = tmp.shape[1]
    tmp = tmp.permute(1,0,2).reshape(-1,1,tmp.shape[-1])
    ans = conv1d(data.view(1,-1,data.shape[-1]), tmp, groups=groups, stride=step)
    return ans.reshape(data.shape[0],tmf_num,-1).permute(1,0,2)



def tmf_super(data,tmp,step=1,device='cpu',moves=[],is_sum=False,batch_size=-1,half=False,save_memory=False):
    if batch_size == -1:
        batch_size = data.shape[0]
    assert device in ['cpu','cuda']
    # savememory模式下，只能使用cuda
    if device == 'cpu' and save_memory:
        print('save_memory mode only support cuda')
        save_memory = False
    if save_memory:
        assert device == 'cuda'
    assert data.ndim == 2
    assert tmp.ndim in [2,3]
    assert data.shape[0] == tmp.shape[1]
    if moves != []:
    # 如果moveout不是numpy类型，转换为numpy类型
        if not isinstance(moves,np.ndarray):
            moves = np.array(moves)
        moveout = [moves[:,i] for i in range(0,moves.shape[1],batch_size)]
        moveout = np.array(moveout).T
    else:
        moveout = []
    if not isinstance(data,torch.Tensor):
        data = torch.from_numpy(data)
    if not isinstance(tmp,torch.Tensor):
        tmp = torch.from_numpy(tmp)
    if tmp.ndim == 2:
        tmp = tmp.unsqueeze(0)
    '''
    if data.dtype != torch.float:
        data = data.float()
    if tmp.dtype != torch.float:
        tmp = tmp.float()
    '''
    if save_memory and half:
        data = data.half().cpu()
        tmp = tmp.half().cpu()
    elif save_memory:
        data = data.float().cpu()
        tmp = tmp.float().cpu()
    else:
        if data.device != device:
            data = data.to(device)
        if tmp.device != device:
            tmp = tmp.to(device)
        if half and device != 'cpu':
            data = data.half()
            tmp = tmp.half()
        else:
          data = data.float()
          tmp = tmp.float()
    data_shape = data.shape
    tmp_shape = tmp.shape
    assert tmp_shape[1] == data_shape[0]
    assert tmp_shape[2] <= data_shape[1]
    assert step > 0
    assert moveout == [] or len(moveout) == tmp_shape[0]
    assert data_shape[0] > 0 and data_shape[1] > 0
    if batch_size > data_shape[0]:
        batch_size = data_shape[0]
    # 对显存爆炸进行监控，如果显存爆炸，就减小batch_size
    while True:
        try:
            for bz in range(0,data_shape[0],batch_size):
                if save_memory and device != 'cpu':
                    tmp_data = data[bz:bz+batch_size,:].cuda()
                    tmp_tmp = tmp[:,bz:bz+batch_size,:].cuda()
                else:
                    tmp_data = data[bz:bz+batch_size,:]
                    tmp_tmp = tmp[:,bz:bz+batch_size,:]
                cc_raw = cc(tmp_data,tmp_tmp,step=step).float()
                norm1 = cc(tmp_data.pow(2),torch.ones_like(tmp_tmp),step=step).float()
                norm2 = tmp_tmp.pow(2).sum(dim=(1,2),keepdims=True).float()
                if half:
                    cc_raw = cc_raw / (norm1 * norm2).sqrt()
                else:
                    cc_raw = cc_raw / (norm1 * norm2).sqrt()
                if save_memory:
                    cc_raw = cc_raw.cpu()
                if bz == 0:
                    cc_raw_all = cc_raw
                else:
                    cc_raw_all = torch.cat([cc_raw_all,cc_raw],dim=1)
            cc_raw = cc_raw_all.cpu().numpy()
            if moveout != []:
                for i in range(tmp_num):
                    move = moveout[i]-moveout[i].min()
                    move = move.astype(int)
                    move = move//step
                    cc_raw[i] = roll(cc_raw[i],move[::-1])
            if is_sum:
                cc_raw = cc_raw.sum(axis=1)
            return cc_raw
        except Exception as e:
            # 清理显存
            try:
                del tmp_data,tmp_tmp,cc_raw,cc_raw_all,norm1,norm2
            except:
                pass
            torch.cuda.empty_cache()
            if batch_size == 1:
                # 提示即便batch_size为1，也无法计算，建议减少其他参数
                print('batch_size is 1, but still can not calculate, please reduce other parameters')
                raise e
            batch_size = batch_size // 2
            print('batch_size is too large, reduce batch_size to {}'.format(batch_size))
            
# test
if __name__ == '__main__':
    try:
        data = torch.randn(100,100)
        tmp = torch.randn(10,100,10)
        cc_raw = tmf_super(data,tmp,step=1,device='cpu',moves=[],is_sum=False,batch_size=1)
    except Exception as e:
        print(e)