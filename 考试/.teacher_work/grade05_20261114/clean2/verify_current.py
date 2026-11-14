import ast, contextlib, hashlib, io, json, os, runpy, sys
import cv2
import numpy as np
from PIL import Image
source = open('test.py', encoding='utf-8').read()
filename = os.path.abspath('test.py')
line = next(n.lineno for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'print')
snapshots = []
def trace(frame, event, arg):
    if event == 'line' and frame.f_code.co_filename == filename and frame.f_lineno == line:
        d = frame.f_locals
        if not snapshots or snapshots[-1][0] != d['angle']:
            snapshots.append((d['angle'], [d[k].copy() for k in ['tv_a','tv_b','cv_a','cv_b']], d['tv_diff'].copy(), d['cv_diff'].copy()))
    return trace
stream = io.StringIO()
sys.settrace(trace)
try:
    with contextlib.redirect_stdout(stream):
        runpy.run_path(filename, run_name='__main__')
finally:
    sys.settrace(None)
assert [x[0] for x in snapshots] == [0, 19], [x[0] for x in snapshots]
image = Image.open('input/compare_practice.png').convert('RGB')
bgr = cv2.imread('input/compare_practice.png', cv2.IMREAD_UNCHANGED)
assert bgr.shape == (211, 337, 3) and bgr.dtype == np.uint8
h, w = bgr.shape[:2]
records=[]
for angle, arrays, td, cd in snapshots:
    tv_a,tv_b,cv_a,cv_b=arrays
    ra = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, center=None, fillcolor=(0,0,0))
    rb = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, center=None, fillcolor=(0,0,0)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    matrix = cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,1.0)
    ca = cv2.warpAffine(bgr[:,::-1,:],matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    cb = cv2.warpAffine(bgr,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))[:,::-1,:]
    refs=[np.array(ra),np.array(rb),ca,cb]
    assert all(np.array_equal(a,r) for a,r in zip(arrays,refs))
    assert all(a.shape==(211,337,3) and a.dtype==np.uint8 for a in arrays)
    assert np.array_equal(td,np.abs(tv_a.astype(np.int16)-tv_b.astype(np.int16)))
    assert np.array_equal(cd,np.abs(cv_a.astype(np.int16)-cv_b.astype(np.int16)))
    assert bool(np.array_equal(tv_a,tv_b)) == (angle==0)
    assert bool(np.array_equal(cv_a,cv_b)) == (angle==0)
    records.append({'angle':angle,'tv_equal':bool(np.array_equal(tv_a,tv_b)),'cv_equal':bool(np.array_equal(cv_a,cv_b)),'tv_mean':float(td.mean()),'cv_mean':float(cd.mean()),'shape':[211,337,3],'dtype':'uint8','reference_all_pixels_match':True,'array_hashes':[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]})
print(json.dumps({'all_checks_passed':True,'stdout':stream.getvalue(),'cases':records,'cv2':cv2.__version__},ensure_ascii=False))
