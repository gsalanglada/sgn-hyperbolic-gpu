import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d

img_path=Path('/mnt/data/conditionA_right_crop.png')
img=cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
# Manual rects detected previously: x,y,w,h in crop coordinates.
# Map to station location. Rectangles include plot frame.
rects = {
    2.0: (60,746,493,199),
    4.0: (62,1036,492,201),
    5.7: (62,1326,493,201),
    10.5:(62,1616,493,199),
    12.5:(62,1904,494,199),
    13.5:(64,2186,493,205),
    14.5:(742,744,489,201),
    15.7:(742,1034,489,203),
    17.3:(742,1324,489,203),
    19.0:(744,1616,489,199),
    21.0:(744,1902,489,201),
}
# y-axis from Dingemans Fig. 3 right panels (condition A): -1.5 to +2.5 cm
Y_MIN, Y_MAX = -1.5, 2.5
T = 2.02
PERIODS = 2.0
n_harm = 3

series = {}
rows=[]
qc=[]
for x_m, (x,y,w,h) in sorted(rects.items()):
    # crop inner plot region. Keep close to edges for peaks, remove frame by 6 px.
    # We avoid the thick border and tick labels by cropping to just inside the frame.
    margin=8
    roi=img[y+margin:y+h-margin, x+margin:x+w-margin]
    H,W=roi.shape
    # Binarize dark line/axes
    mask = roi < 170
    # remove rows/cols that are essentially axes/ticks: long horizontal zero line and border remnants
    row_count = mask.sum(axis=1)
    # zero line is the dominant row near the middle; top/bottom borders are already cropped out.
    # zero line is expected near eta=0, about 5/8 of the way from the top
    lo=int(0.35*H); hi=int(0.80*H)
    y_zero = lo + int(np.argmax(row_count[lo:hi]))
    if row_count[y_zero] > 0.35*W:
        mask[max(0,y_zero-4):min(H,y_zero+5),:] = False
    # Remove vertical tick/axis remnants near left/right edges and very top/bottom tick marks
    mask[:, :3] = False
    mask[:, -3:] = False
    mask[:10, :] = False
    mask[-8:, :] = False
    # component filtering: drop tiny ticks/noise; keep components broad enough to be data curve
    num, labels, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=bool)
    for lab in range(1,num):
        xx,yy,ww,hh,area = stats[lab]
        if area < 10:
            continue
        # drop components that are pure tick marks or text fragments
        if ww < 15:
            continue
        # drop top/bottom tick bars detached from the curve
        if hh < 8 and ww < 80:
            continue
        keep |= (labels==lab)
    mask=keep
    xs=[]; ys=[]
    prev_y=None
    for j in range(W):
        yy=np.where(mask[:,j])[0]
        if len(yy)>0:
            # Use curve-continuity to avoid vertical tick marks when several dark pixels
            # are present in the same column.
            if prev_y is None:
                val=float(np.median(yy))
            else:
                val=float(yy[np.argmin(np.abs(yy-prev_y))])
            xs.append(j); ys.append(val); prev_y=val
        else:
            # keep previous y for continuity across short gaps
            pass
    xs=np.array(xs); ys=np.array(ys)
    # If there are gaps, interpolate over full width.
    if len(xs)<0.25*W:
        print('WARN few points', x_m, len(xs), 'of', W)
    # Remove isolated outliers by sliding median on y if needed
    if len(xs)>10:
        # interpolate to all pixels first
        f=interp1d(xs,ys,bounds_error=False,fill_value='extrapolate')
        allx=np.arange(W)
        ypix=f(allx)
    else:
        allx=np.arange(W)
        ypix=np.full(W,np.nan)
    # Convert pixels to cm: top of inner crop corresponds approximately Y_MAX, bottom to Y_MIN.
    # border crop shrinks the axis: adjust using the original plot frame interior height h-2*margin.
    eta = Y_MAX - (ypix / (H-1)) * (Y_MAX - Y_MIN)
    t = np.linspace(0, PERIODS*T, W)
    # remove possible NaN
    ok=np.isfinite(eta)
    # Fit constant + harmonics on 2 full periods.
    tt=t[ok]; yy=eta[ok]
    cols=[np.ones_like(tt)]
    for n in range(1,n_harm+1):
        cols += [np.cos(2*np.pi*n*tt/T), np.sin(2*np.pi*n*tt/T)]
    A=np.vstack(cols).T
    coeff, *_=np.linalg.lstsq(A, yy, rcond=None)
    amps=[]
    for n in range(n_harm):
        c=coeff[1+2*n]
        s=coeff[2+2*n]
        amps.append(float(np.sqrt(c*c+s*s)))
    rows.append({'x_m':x_m,'A1_cm':amps[0],'A2_cm':amps[1],'A3_cm':amps[2],
                 'mean_eta_cm':float(coeff[0]),'source':'Dingemans1994 Fig.3 right-panel digitization; condition A; approx.'})
    for ti,ei in zip(t,eta):
        series.setdefault('time_s',t)
    series[f'eta_x{x_m:g}_cm']=eta
    qc.append((x_m,len(xs),W,y_zero,row_count[y_zero]))

# Save harmonics
out_h=Path('/mnt/data/beji_lab_harmonics.csv')
df=pd.DataFrame(rows).sort_values('x_m')
df.to_csv(out_h,index=False,float_format='%.6g')
# Save digitized time series in wide and long forms.
out_ts=Path('/mnt/data/beji_lab_conditionA_timeseries_digitized.csv')
common_t=np.linspace(0, PERIODS*T, 500)
ts=pd.DataFrame({'time_s':common_t})
for x_m in sorted(rects):
    eta=series[f'eta_x{x_m:g}_cm']
    t_local=np.linspace(0, PERIODS*T, len(eta))
    ts[f'eta_x{x_m:g}_cm']=np.interp(common_t,t_local,eta)
ts.to_csv(out_ts,index=False,float_format='%.6g')
# Long form with native extraction resolution
long=[]
for x_m in sorted(rects):
    eta=series[f'eta_x{x_m:g}_cm']
    t=np.linspace(0, PERIODS*T, len(eta))
    for ti,ei in zip(t,eta): long.append({'x_m':x_m,'time_s':ti,'eta_cm':ei})
pd.DataFrame(long).to_csv('/mnt/data/beji_lab_conditionA_timeseries_digitized_long.csv',index=False,float_format='%.6g')
# Write a QC overlay image: plot digitized points over the original crop in red.
qc_img=Image.open(img_path).convert('RGB')
draw=ImageDraw.Draw(qc_img)
for x_m,(x,y,w,h) in sorted(rects.items()):
    margin=8
    eta=series[f'eta_x{x_m:g}_cm']
    H2=h-2*margin; W2=w-2*margin
    pts=[]
    for j,e in enumerate(eta):
        # map eta to pixel in inner crop
        yp=(Y_MAX-e)/(Y_MAX-Y_MIN)*(H2-1)
        xp=j/(len(eta)-1)*(W2-1)
        pts.append((x+margin+xp,y+margin+yp))
    # draw every few segments to avoid huge memory
    draw.line(pts,fill=(255,0,0),width=2)
qc_img.save('/mnt/data/beji_conditionA_digitization_QC.png')
print(out_h)
print(out_ts)
print(df)
print('QC', qc)
