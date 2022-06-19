# stepA
factor_stepA_scale = 1000# 第一个阶段先把图像放缩到较小，提高计算效率
num_match_points = 80 #至少与模板匹配到X个点
num_choice_point = 500 #定位图片的时候，抽取x对点，计算文字成比例程度
num_filter = 20 #可能偶尔有几个匹配错误的点（偏离均值比较远），可以过滤掉
th_var_error = 5 #角度的方差
cut_extend = 0.1 # 多裁出来一些边框，是长度和宽度的比例
rotate_angle = 3 #如果小于X度就不转了，效率低，也不一定真的正
th_error_rect = 15 #矩形程度误差至少要小于该值（四个角偏离直角的度数累加）
min_h_or_w = 480 # 舍弃尺寸较小的照片
extend_pix = 0 # 粗略判断框体在图像内部，如果在外部可以有X个像素的容错

score_out_of_pic = 9999

h_output = 371#337 #最终输出的证件高度
w_output = 600#532

def remove_portrait(img):
    # img[35:265, 328:507] = 0  # 抠掉头像
    img[int(h_output * 0.07):int(h_output * 0.82), int(w_output * 0.59):int(w_output * 0.96)] = 0
    return img

# stepB
# dic_tmp_factor_w_h = {"box.png":1.6,"box_meng.png":1.72}

# SIFT模板存放的文件夹
dir_tmp = "pic_tmps/"

#如果输入图片过小，将图片拷贝到该文件夹
dir_small_input = "dir_small_input/"

# 模糊
dir_vague = "dir_vague/"

# 畸变
dir_distortion = "dir_distortion/"

# 缺失
dir_incomplete = "dir_incomplete/"

# 正常的输出
dir_output = "dir_output/"