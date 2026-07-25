import os, pickle, sys, numpy as np, mujoco, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
D=os.path.dirname(__file__); tag=sys.argv[1] if len(sys.argv)>1 else "square"
XML=os.path.join(D,"ant.xml"); mj=mujoco.MjModel.from_xml_path(XML)
GT=np.array([mj.geom_type[g] for g in range(mj.ngeom)]); GS=np.array([mj.geom_size[g] for g in range(mj.ngeom)])
log=pickle.load(open(f"{D}/sgfollow_{tag}.pkl","rb")); wps=np.load(f"{D}/sgfollow_{tag}_wps.npy")
xs=[l[2][0] for l in log]; ys=[l[2][1] for l in log]; imgs=[]
for fi in range(0,len(log),4):
    gx,gm,pos,yaw,gi,dist,rel=log[fi]
    fig=plt.figure(figsize=(9.2,4.5))
    ax=fig.add_subplot(1,2,1,projection='3d')
    for g in range(mj.ngeom):
        if GT[g]==3:
            hl=GS[g][1]; a=gx[g]-gm[g][:,2]*hl; b=gx[g]+gm[g][:,2]*hl
            ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],c='#1f5fa8',lw=4)
        elif GT[g]==2: ax.scatter([gx[g][0]],[gx[g][1]],[gx[g][2]],s=240,c='#d9541e')
    c=gx[1]; ax.set_xlim(c[0]-1.2,c[0]+1.2); ax.set_ylim(c[1]-1.2,c[1]+1.2); ax.set_zlim(0,1.2); ax.set_box_aspect([abs(ax.get_xlim()[1]-ax.get_xlim()[0]),abs(ax.get_ylim()[1]-ax.get_ylim()[0]),abs(ax.get_zlim()[1]-ax.get_zlim()[0])])
    ax.view_init(elev=26,azim=-62); ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_zticklabels([])
    ax.set_title(f"MuJoCo Ant (skid-steer)  step {fi}")
    ax2=fig.add_subplot(1,2,2)
    # subgoal path (dashed) + numbered subgoals
    wpc=np.vstack([[xs[0],ys[0]],wps]); ax2.plot(wpc[:,0],wpc[:,1],'--',c='gray',lw=1,alpha=0.6)
    for j,w in enumerate(wps):
        col='green' if j<gi else ('#e0b000' if j==gi else 'lightgray')
        ax2.scatter([w[0]],[w[1]],marker='*',s=260,c=col,edgecolor='k',lw=0.5,zorder=5)
        ax2.annotate(f"sg{j+1}",(w[0],w[1]),textcoords="offset points",xytext=(6,6),fontsize=8)
    ax2.plot(xs[:fi+1],ys[:fi+1],'-',c='#1f5fa8',lw=2)
    ax2.scatter([pos[0]],[pos[1]],c='#d9541e',s=70,zorder=6,edgecolor='k',lw=0.5)
    ax2.arrow(pos[0],pos[1],0.45*np.cos(yaw),0.45*np.sin(yaw),head_width=0.14,color='#d9541e',zorder=7)
    ax2.set_xlim(0,6);ax2.set_ylim(0.5,5.5);ax2.set_aspect('equal');ax2.grid(alpha=.3)
    ax2.set_title(f"Following subgoals | target=sg{gi+1}  dist={dist:.2f}  relax={rel:.3f}")
    fig.tight_layout(); fig.canvas.draw()
    img=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1]+(4,))[...,:3]
    imgs.append(img.copy()); plt.close(fig)
pil=[Image.fromarray(im) for im in imgs]; path=f"{D}/gifs/ant_subgoal_follow_{tag}.gif"
pil[0].save(path,save_all=True,append_images=pil[1:],duration=60,loop=0,optimize=True)
print("wrote",path,len(imgs),"frames",os.path.getsize(path)//1024,"KB")
Image.fromarray(imgs[len(imgs)//2]).save(f"{D}/sgf_mid.png"); Image.fromarray(imgs[-1]).save(f"{D}/sgf_last.png")
