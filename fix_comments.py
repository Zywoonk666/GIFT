import os
import re

filepath = '/root/GIFT/model/GIFT/GIMLETEncoderStack.py'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# Remove t-SNE block
text = re.sub(r'# --- \[Start\] 保存 OT 对齐特征供 t-SNE 可视化 ---.*?# --- \[End\] ---', '', text, flags=re.DOTALL)

replacements = {
    '计算文本 Token 和图基序之间的最优传输分配计划 T。': 'Compute optimal transport plan T.',
    '参数:': 'Args:',
    '文本特征 H_t': 'Text features H_t',
    '图特征 H_m': 'Graph features H_m',
    '熵正则化系数 (控制分配矩阵的平滑度，越小越精确但也越难收敛)': 'Entropy regularization coefficient (controls smoothness, smaller is harder to converge)',
    'Sinkhorn 迭代次数': 'Sinkhorn iteration times',
    '返回:': 'Returns:',
    '最优分配计划矩阵，shape': 'Optimal transport assignment plan matrix, shape',
    '步骤 1：计算局部的传输成本矩阵 C (Cost Matrix)': 'Step 1: Compute local transport cost matrix C',
    '对特征在最后一个维度 (Dim) 上进行 L2 归一化': 'L2 normalize features on the last dimension',
    '通过矩阵乘法计算余弦相似度:': 'Compute cosine similarity via matrix multiplication:',
    '结果 shape:': 'Result shape:',
    '计算成本矩阵:': 'Compute cost matrix:',
    '文本端分布': 'Text side distribution',
    '图像端分布': 'Graph side distribution',
    '步骤 2：使用 Sinkhorn 算法求解分配计划 T': 'Step 2: Use Sinkhorn algorithm to solve for T',
    '计算 Kernel 矩阵:': 'Compute Kernel matrix:',
    '初始化双对偶变量': 'Initialize dual variables',
    '迭代寻找最优解': 'iterations to find optimal solution',
    '更 新': 'Update',     
    '更新 ': 'Update ',
    '计算最终的最优传输分配矩阵 T': 'Compute final optimal transport assignment matrix T',
    '带残差连接的 MLP 模块，增加 Dropout 防止小数据集过拟合。': 'MLP module with residual connection, adding Dropout to prevent overfitting.',
    '增加 dropout 参数': 'Add dropout parameter',
    '初始化 Dropout': 'Initialize Dropout',
    '在激活函数后添加 Dropout': 'Add Dropout after activation function',
    '由多个残差模块堆叠而成的深层 Projector。': 'Deep Projector stacked by multiple residual modules.',
    '堆叠多个模块': 'Stack multiple modules',
    '输入和输出的特征维度': 'Feature dimension of input and output',
    '残差块的数量 (2个 block 相当于 4层 Linear)': 'Number of residual blocks (2 blocks equal 4 linear layers)',
    '激活函数类型': 'Type of activation function',
    '关闭 渐进式解耦，减小掩码影响': 'Disable progressive decoupling, reduce mask impact',
    '是否使用独立的图特征 QKV 和 FFN 专家层': 'Whether to use independent graph feature QKV and FFN expert layers',
    '仅仅替换 encoder layer 的核心内容': 'Only replace the core content of encoder layer',
    '未掩码': 'Unmasked',
    '严格掩码': 'Strictly masked',
    '训练阶段：按概率 mask_ratio 生成随机掩码矩阵。': 'Training phase: Generate random mask matrix according to probability mask_ratio.',
    '如果 ratio>=1.0 或为测试模式，则 100% 执行严格跨模态掩码': 'If ratio>=1.0 or in test mode, 100% execute strict cross-modal mask',
    '假设总共有 12 层，例如可以让前 8 层(0-7)保持掩码，后 4 层(8-11)放开掩码': 'Assuming 12 layers in total, e.g. keep first 8 layers masked, and unmask last 4 layers',
    '传入 graph_length': 'Pass graph_length',
    '图专家层专属参数': 'Graph expert layer specific parameters',
    '的特征维度': 'features dimension',
    '最优传输分配计划 T': 'optimal transport assignment plan T',
    '和': 'and'
}

for k, v in replacements.items():
    text = text.replace(k, v)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)
print("Done")
