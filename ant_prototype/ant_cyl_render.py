import os, pickle, numpy as np, mujoco, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
D=os.path.dirname(__file__); RCYL=0.90; AREA=12.0; N=3
goals=np.array([[10.,9.],[10.,6.],[10.,3.]]); COL=['#1f5fa8','#d9541e','#2ca25f']
XML=os.path.join(D,"ant.xml"); mj=mujoco.MjModel.from_xml_path(XML)
GT=np.array([mj.geom_type[g] for g in range(mj.ngeom)]); GS=np.array([mj.geom_size[g] for g in range(mj.ngeom)])
log=pickle.load(open(f"{D}/cyl_multi_log.pkl","rb"))
paths=[[] for _ in range(N)]
for l in log:
    for i in range(N): paths[i].append(l[1][i])
imgs=[]
for fi in range(0,len(log),5):
    geoms,pos,dists,heads,rel,pd=log[fi]
    fig=plt.figure(figsize=(9.8,4.6))
    ax=fig.add_subplot(1,2,1,projection='3d')
    for i in range(N):
        gx,gm,off=geoms[i]
        for g in range(mj.ngeom):
            if GT[g]==3:
                hl=GS[g][1]; a=gx[g]-gm[g][:,2]*hl; b=gx[g]+gm[g][:,2]*hl
                ax.plot([a[0]+off[0],b[0]+off[0]],[a[1]+off[1],b[1]+off[1]],[a[2],b[2]],c=COL[i],lw=2.5)
            elif GT[g]==2: ax.scatter([gx[g][0]+off[0]],[gx[g][1]+off[1]],[gx[g][2]],s=90,c=COL[i])
    ax.set_xlim(1,11);ax.set_ylim(1,11);ax.set_zlim(0,1.0);ax.view_init(elev=40,azim=-70); ax.set_box_aspect([abs(ax.get_xlim()[1]-ax.get_xlim()[0]),abs(ax.get_ylim()[1]-ax.get_ylim()[0]),abs(ax.get_zlim()[1]-ax.get_zlim()[0])])
    ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_zticklabels([]); ax.set_title(f"3 Ants (r=0.90 cyl)  step {fi}")
    ax2=fig.add_subplot(1,2,2)
    for i in range(N):
        p=np.array(paths[i][:fi+1])
        if len(p): ax2.plot(p[:,0],p[:,1],'-',c=COL[i],lw=1.5,alpha=0.7)
        cyl=plt.Circle(pos[i],RCYL,color=COL[i],alpha=0.18,zorder=2); ax2.add_patch(cyl)     # 0.90 safety cylinder
        ax2.add_patch(plt.Circle(pos[i],RCYL,fill=False,ec=COL[i],lw=1.2,zorder=3))
        ax2.scatter([pos[i][0]],[pos[i][1]],c=COL[i],s=45,zorder=5,edgecolor='k',lw=0.4)
        h=heads[i]; ax2.arrow(pos[i][0],pos[i][1],0.6*h[0],0.6*h[1],head_width=0.22,color=COL[i],zorder=6)
        ax2.scatter([goals[i,0]],[goals[i,1]],marker='*',c=COL[i],s=200,zorder=5,edgecolor='k',lw=0.4)
    warn = pd < 2*RCYL
    ax2.set_xlim(0,12);ax2.set_ylim(0,12);ax2.set_aspect('equal');ax2.grid(alpha=.3)
    ax2.set_title(f"min gap={pd:.2f}m (safe≥{2*RCYL:.1f})  relax={rel:.2f}",
                  color=('red' if warn else 'black'))
    fig.tight_layout(); fig.canvas.draw()
    img=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1]+(4,))[...,:3]
    imgs.append(img.copy()); plt.close(fig)
pil=[Image.fromarray(im) for im in imgs]; path=f"{D}/gifs/ant_cylinder_multi.gif"
pil[0].save(path,save_all=True,append_images=pil[1:],duration=60,loop=0,optimize=True)
print("wrote",path,len(imgs),"frames",os.path.getsize(path)//1024,"KB")
# frame at the tightest moment
mins=[l[5] for l in log]; ti=int(np.argmin(mins))//5
Image.fromarray(imgs[min(ti,len(imgs)-1)]).save(f"{D}/cyl_tight.png"); Image.fromarray(imgs[-1]).save(f"{D}/cyl_last.png")
print("tightest gap frame saved, min gap=",min(mins))
