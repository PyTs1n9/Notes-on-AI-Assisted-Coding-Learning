
import contextlib, io, runpy
with contextlib.redirect_stdout(io.StringIO()):
    d = runpy.run_path('test.py')
import torch
expected = torch.arange(1,25,dtype=torch.float32).reshape(6,1,2,2)
y = torch.tensor([1,0,1,0,1,0],dtype=torch.int64)
ti=[0,2,3,5]; vi=[4,1]
a=expected[ti]; mean=a.mean(); std=a.std(unbiased=False)
assert torch.equal(d['images'],expected)
assert torch.equal(d['labels'],y)
assert mean.item()==12.5
assert torch.equal(d['train_mean'],mean) and torch.equal(d['train_std'],std)
for name, inds in [('train',ti),('val',vi)]:
    ds=d[name+'_dataset']; assert len(ds)==len(inds)
    assert torch.equal(ds.images,expected[inds]) and torch.equal(ds.labels,y[inds])
    assert torch.equal(ds.mean,mean) and torch.equal(ds.std,std)
    for i,original in enumerate(inds):
        im,lab=ds[i]
        assert im.shape==(1,2,2) and im.dtype==torch.float32 and im.device.type=='cpu'
        assert lab.dtype==torch.int64 and lab.shape==torch.Size([])
        assert torch.equal(im,(expected[original]-mean)/std) and lab==y[original]
tl=d['train_loader']; vl=d['val_loader']
assert tl.batch_size==3 and tl.drop_last and tl.num_workers==0
assert isinstance(tl.sampler,torch.utils.data.RandomSampler)
assert vl.batch_size==2 and not vl.drop_last and vl.num_workers==0
assert isinstance(vl.sampler,torch.utils.data.SequentialSampler)
assert len(tl)==len(vl)==1
records=[]
for seed in [7,21]:
    torch.manual_seed(seed); kept=[]
    for im,lab in tl:
        assert im.shape==(3,1,2,2) and lab.shape==(3,)
        for image,label in zip(im,lab):
            found=[]
            for j in ti:
                if torch.equal(image,(expected[j]-mean)/std): found.append(j)
            assert len(found)==1 and label==y[found[0]]
            kept.append(found[0])
    assert len(kept)==len(set(kept))==3
    dropped=sorted(set(ti)-set(kept)); records.append((seed,kept,dropped))
for im,lab in vl:
    assert torch.equal(im,(expected[vi]-mean)/std) and torch.equal(lab,y[vi])
assert any(r[2] != [ti[-1]] for r in records)
print('PASS: 全图像/标签、互斥索引、训练统计量、单样本、加载器参数、全部批次与身份追踪')
print('实际采样身份',records)
