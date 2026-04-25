import matplotlib
matplotlib.use('Agg')  # 必须在 import matplotlib.pyplot 之前设置后端
import matplotlib.pyplot as plt

def setup_matplotlib_style():
    """全局配置 matplotlib 中文显示和无界面后端"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False