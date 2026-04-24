import matplotlib.pyplot as plt
import matplotlib

def setup_matplotlib_style():
    """全局配置 matplotlib 中文显示和无界面后端"""
    matplotlib.use('Agg')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False