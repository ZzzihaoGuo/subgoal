import os, pickle, numpy as np, mujoco, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
D=os.path.dirname(__file__)
XML=os.path.join(os.path.dirname(__file__),"ant.xml")
mj=mujoco.MjModel.from_xml_path(XML)
GT=np.array([mj.geom_type[g] for g in range(mj.ngeom)]); GS=np.array([mj.geom_size[g] for g in range(mj.ngeom)])
meta=np.load(f"{D}/manifold_log_meta.npy")            # x,y,yaw,dist,relax
geoms=pickle.load(open(f"{D}/manifold_geoms.pkl","rb"))
obs=np.load(f"{D}/manifold_obs.npy")                  # (4,2) corners
goal=np.array([5.0,3.0]); off=meta[0,:2]-geoms[0][0][1][:2]  # env offset for 3D->env
imgs=[]; xs=meta[:,0]; ys=meta[:,1]
for i in range(0,len(geoms),3):
    gx,gm=geoms[i]; mx,my,yaw,dist,rel=meta[i]
    fig=plt.figure(figsize=(9,4.4))
    ax=fig.add_subplot(1,2,1,projection='3d')
    for g in range(mj.ngeom):
        if GT[g]==3:
            hl=GS[g][1]; a=gx[g]-gm[g][:,2]*hl; b=gx[g]+gm[g][:,2]*hl
            ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],c='#1f5fa8',lw=4)
        elif GT[g]==2:
            ax.scatter([gx[g][0]],[gx[g][1]],[gx[g][2]],s=240,c='#d9541e')
    c=gx[1]; ax.set_xlim(c[0]-1.2,c[0]+1.2); ax.set_ylim(c[1]-1.2,c[1]+1.2); ax.set_zlim(0,1.2); ax.set_box_aspect([abs(ax.get_xlim()[1]-ax.get_xlim()[0]),abs(ax.get_ylim()[1]-ax.get_ylim()[0]),abs(ax.get_zlim()[1]-ax.get_zlim()[0])])
    ax.view_init(elev=22,azim=-60); ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_zticklabels([])
    ax.set_title(f"MuJoCo Ant (real dynamics)  step {i}")
    ax2=fig.add_subplot(1,2,2)
    ax2.add_patch(Polygon(obs, closed=True, color='#c0392b', alpha=0.55, label='obstacle'))
    ax2.plot(xs[:i+1],ys[:i+1],'-',c='#1f5fa8',lw=2,label='CoM path')
    ax2.scatter([mx],[my],c='#d9541e',s=70,zorder=5)
    ax2.arrow(mx,my,0.4*np.cos(yaw),0.4*np.sin(yaw),head_width=0.12,color='#d9541e',zorder=6)
    ax2.scatter([goal[0]],[goal[1]],marker='*',c='green',s=220,label='goal',zorder=5)
    ax2.set_xlim(0,6);ax2.set_ylim(1.2,4.8);ax2.set_aspect('equal');ax2.grid(alpha=.3)
    ax2.legend(loc='lower right',fontsize=8)
    ax2.set_title(f"Nav space | dist2goal={dist:.2f}  manifold relax={rel:.3f}")
    fig.tight_layout(); fig.canvas.draw()
    img=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1]+(4,))[...,:3]
    imgs.append(img.copy()); plt.close(fig)
pil=[Image.fromarray(im) for im in imgs]
path=f"{D}/ant_manifold.gif"
pil[0].save(path, save_all=True, append_images=pil[1:], duration=66, loop=0, optimize=True)
print("wrote",path,len(imgs),"frames",os.path.getsize(path)//1024,"KB")
Image.fromarray(imgs[len(imgs)//2]).save(f"{D}/mf_mid.png"); Image.fromarray(imgs[-1]).save(f"{D}/mf_last.png")
