import matplotlib.pyplot as plt
import numpy as np

models = ["V-JEPA", "VideoMAE-v2", "gemini-2.5pro", "Qwen3.7-plus"]
O1 = [100, 55, 80.83, 88.15]
O2 = [100, 65, 81, 77.5]
O3 = [95, 58.33, 81.82, 80]

x = np.arange(len(models))
width = 0.25

# 顶级期刊经典三色（深蓝 / 翠绿 / 暖橙）高级耐看
c1 = '#194775'
c2 = '#279365'
c3 = '#e67e22'

fig, ax = plt.subplots(figsize=(10,6), dpi=150)
b1 = ax.bar(x-width, O1, width, label='O1', color=c1, edgecolor='white', linewidth=0.6)
b2 = ax.bar(x,      O2, width, label='O2', color=c2, edgecolor='white', linewidth=0.6)
b3 = ax.bar(x+width, O3, width, label='O3', color=c3, edgecolor='white', linewidth=0.6)

ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Model Accuracy Comparison', fontsize=12, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=18, fontsize=9)
ax.legend(loc='lower right', frameon=False, fontsize=9)
ax.set_ylim(0, 108)
ax.grid(axis='y', alpha=0.2, linestyle='-')

# 隐藏上右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 标注数值
def add_label(bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1.2, f'{h:.1f}', ha='center', fontsize=7.5)

add_label(b1)
add_label(b2)
add_label(b3)

plt.tight_layout()
plt.savefig('acc_compare.png', dpi=300, bbox_inches='tight')
plt.show()