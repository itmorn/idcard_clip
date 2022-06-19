import numpy as np

def GetClockAngle(v1, v2):
    # v1顺时针旋转到v2的角度
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
    # 叉乘
    rho =  np.rad2deg(np.arcsin(np.clip(np.cross(v1, v2)/TheNorm,-1,1)))
    # 点乘
    theta = np.rad2deg(np.arccos(np.clip(np.dot(v1,v2)/TheNorm,-1,1)))
    if rho < 0:
        return  theta
    else:
        return 360-theta

