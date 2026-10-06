from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = r'C:\Users\lin\Desktop\Q.docx'

def u(s):
    return s.encode('ascii').decode('unicode_escape') if s.isascii() else s

def font_style(style, size, color=None):
    style.font.name = 'Microsoft YaHei'
    style._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    style.font.size = Pt(size)
    if color:
        style.font.color.rgb = RGBColor.from_string(color)

def shade(cell):
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), 'D9EAF7')
    cell._tc.get_or_add_tcPr().append(shd)

def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, value in enumerate(headers):
        table.rows[0].cells[i].text = value
        shade(table.rows[0].cells[i])
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.name = 'Microsoft YaHei'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
                    run.font.size = Pt(8.5)
    doc.add_paragraph()

def bullet(doc, text):
    doc.add_paragraph(text, style='List Bullet')

doc = Document()
section = doc.sections[0]
section.top_margin = Inches(.65)
section.bottom_margin = Inches(.65)
section.left_margin = Inches(.75)
section.right_margin = Inches(.75)
font_style(doc.styles['Normal'], 9.5)
font_style(doc.styles['Title'], 20, '17365D')
font_style(doc.styles['Heading 1'], 14, '1F4E79')
font_style(doc.styles['Heading 2'], 11, '2F75B2')

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Q\n' + u('跌倒检测 Crossbar-CNN-SNN 工程文档'))
r.bold = True
r.font.size = Pt(20)
r.font.color.rgb = RGBColor(23, 54, 93)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run(u('基准版本：v8 p006 / v8-final    文档日期：2026-07-29')).italic = True

doc.add_heading(u('1. 文档范围与基线定义'), 1)
doc.add_paragraph(u('本工程面向视频帧序列中的 ADL（Activities of Daily Living）与 Fall 二分类识别，当前正式基线为 v8 p006。v8 p006 在 v7 CNN-GRU-SNN group-channel 基础上加入归一化 fall_event_score、阈值触发的 SNN fusion gate 以及 ADL hard-negative gate penalty，目标是在保持 Fall recall 的同时降低 ADL 误报。'))
bullet(doc, u('正式模型标签：p006；snn_adl_gate_penalty=0.06；验证集选择阈值=0.70。'))
bullet(doc, u('Subject4 只用于最终独立测试，不参与模型、阈值、采样权重或 penalty 选择。'))
bullet(doc, u('后续 fusion-gate 扫描属于探索实验，不能覆盖或替代 v8-final。'))

doc.add_heading(u('2. 技术背景'), 1)
doc.add_paragraph(u('跌倒识别同时包含空间外观、短时运动变化和跌倒后静止/低姿态等时间上下文。普通 CNN 擅长提取单帧空间特征，但对重心下移、姿态变化、快速运动和落地后保持的顺序关系表达有限。因此本项目采用 CNN-GRU 主分支与轻量 SNN temporal branch 并行：CNN 提取空间表征，GRU 聚合 16 帧顺序，SNN 将运动、重心、姿态和事件特征转换为脉冲式时序表征，最后通过 fusion gate 调节 SNN 分支贡献。'))
doc.add_paragraph(u('工程还模拟忆阻器/交叉阵列部署条件：图像映射到 Crossbar 读出，加载 HRS/LRS 电导数据，并将 EPSC、PPF、LTP、LTD 实测响应曲线用于器件动态扰动和训练噪声建模。'))

doc.add_heading(u('3. 项目文件与路径'), 1)
add_table(doc, [u('类别'), u('路径'), u('用途')], [
    (u('正式工程目录'), r'F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8', u('v8 p006 主工程、模型、报告和图像')),
    (u('核心代码'), r'F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\code\automatic_fall_detector.py', u('数据、模型、训练、验证、推理和器件建模主入口')),
    (u('正式模型'), r'F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth', u('v8 p006 最终 checkpoint')),
    (u('正式配置'), r'F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\v8_final_config_p006.json', u('版本、penalty、阈值、环境和数据路径')),
    (u('器件数据'), r'F:\12.18-2\data', u('HRS/LRS、EPSC、PPF、LTP、LTD')),
    (u('训练数据'), r'F:\12.18-2\测试集\13354453\...\Subject 1、Subject 2；F:\12.18-2\zenodo_falldb_video_split\train', u('GMDCSA Subject1-2 + Zenodo train')),
    (u('验证数据'), r'F:\12.18-2\测试集\13354453\...\Subject 3；F:\12.18-2\zenodo_falldb_video_split\val', u('GMDCSA Subject3 + Zenodo val')),
    (u('最终测试'), r'F:\12.18-2\gmdcsa_subject4_test\Subject 4', u('最终独立 Subject4 测试')),
    (u('论文图表'), r'F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\ppt_v8_final', u('流程图、网络图、噪声图、识别案例和图表数据')),
])

doc.add_heading(u('4. 算法路线'), 1)
for text in [
    u('数据读取：按视频生成 16 帧时间窗口，训练 stride=16；GMDCSA 使用 CSV 帧级 Fall 标注，Zenodo 使用目录/视频标签。'),
    u('预处理：灰度化、缩放到 64×64；训练集采用 teacher-student Gaussian noise。'),
    u('器件映射：图像进入 Crossbar 近似读出，HRS/LRS 用于电导映射，EPSC/PPF/LTP/LTD 用于动态响应。'),
    u('CNN 编码：Conv(1→16) → Conv(16→32) → Conv(32→64) → Conv(64→96)，配合 BatchNorm、ReLU、池化和 Adaptive Average Pooling(2×2)，最后 Linear 384→64。'),
    u('GRU 时序：64 维帧特征输入单层双向 GRU，hidden size=64；mean/max/last 三种池化拼接成主分支表示。'),
    u('SNN 时序：提取 mass、motion、center_y、center_drop、soft_height、aspect、fall_event_like 七类特征，进入可学习 decay/threshold 的 LIF-inspired branch。'),
    u('Fusion gate：p006 使用 group-channel gate，group gate 产生 3 个事件组权重，channel gate 产生 96 个通道权重。'),
    u('分类头：主分支与 SNN 分支拼接后进入 Linear → ReLU → Dropout(0.25) → Linear。'),
    u('训练策略：从 v7 group-channel checkpoint 初始化，前 2 个 epoch 冻结 CNN/GRU，之后解冻，并配合 hard-negative sampler 和 early stopping。'),
]: bullet(doc, text)

doc.add_heading(u('5. v8 p006 核心参数与结果'), 1)
add_table(doc, [u('项目'), u('设置')], [
    (u('输入 / 时间窗'), '64×64；16 frames'),
    (u('hidden_dim / SNN hidden'), '64 / 32'),
    (u('SNN dynamics'), 'decay=0.82；threshold=0.55；learnable=True'),
    (u('fusion gate'), 'group_channel；event threshold=0.55；sharpness=10'),
    (u('ADL penalty'), 'snn_adl_gate_penalty=0.06；normal_positive_penalty=0'),
    (u('hard-negative'), 'weight=1.10；percentile=0.88；gamma=2.0；max multiplier=4.0'),
    (u('数据规模'), 'train=3539（ADL 1758, Fall 1781）；val=1076（ADL 573, Fall 503）'),
])
add_table(doc, [u('结果口径'), 'Accuracy', 'Balanced Acc.', 'ADL specificity', 'Fall recall', u('说明')], [
    (u('p006 strict val，threshold 0.70'), '72.13%', '72.04%', '66.67%', '77.42%', u('Subject3 + Zenodo val')),
    (u('p006 Subject4，threshold 0.70'), '78.38%', '78.24%', '80.00%', '76.47%', u('最终独立测试；TP/TN/FP/FN=13/16/4/4')),
])

doc.add_heading(u('6. 当前项目进度与问题'), 1)
for text in [
    u('v8 p006 正式基线已完成，模型、配置、验证摘要和 Subject4 独立测试均已保存。'),
    u('噪声鲁棒性评估已完成，可复用 Gaussian、Poisson、Salt-Pepper 等多噪声重复实验输出。'),
    u('fusion gate floor=0.35 已完成：最佳 epoch4，Bal Acc 73.2%、ADL specificity 65.8%、Fall recall 80.5%。'),
    u('fusion gate floor=0.50 已完成：最佳 epoch4，Bal Acc 73.2%、ADL specificity 65.6%、Fall recall 80.7%。两组尚未超过 p006。'),
    u('主要问题是 group/channel 双重 gate 初始抑制偏强，可能压制有效 SNN 动态通道；同时候选时间特征主要依靠启发式规则。'),
    u('SNN 加入后最佳结果常出现在 epoch1，说明新增分支或 gate 在后续联合解冻时可能扰动已训练好的 CNN-GRU 表征。'),
]: bullet(doc, text)

doc.add_heading(u('7. 建议的下一步路线'), 1)
for text in [
    u('固定 p006 数据、初始化和器件配置，仅扫描 group-only、group-channel、residual floor=0.20/0.35/0.50。'),
    u('优先测试弱抑制 gate，而不是继续增大 penalty；让 group gate 主导，channel gate 只做小幅修正。'),
    u('增加 temporal auxiliary head，监督 impact、center drop、posture change、ground hold 四类时间目标。'),
    u('采用冻结 CNN/GRU 2轮、仅解冻 GRU、最后再考虑全量微调的分阶段策略。'),
    u('建立 ADL hard negative 与 Fall positive 错误案例库，分析 gate 和 event score 的时间曲线。'),
    u('Subject4 只在方案冻结后做一次最终测试，不能用于选通道、阈值、penalty 或学习率。'),
]: bullet(doc, text)

doc.add_heading(u('8. 复现实验检查清单'), 1)
for text in [
    r'Python：F:\ANACONDA\envs\my_yizu_3.10\python.exe',
    r'主代码：F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\code\automatic_fall_detector.py',
    u('训练窗口：3539；验证窗口：1076；image-size=64；window-size=16；external-labeled-stride=16。'),
    u('日志必须出现：Loaded conductance data、Loaded device dynamics、external_labeled_windows=3539、external_val_labeled_windows=1076。'),
    r'正式模型：F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth',
    u('不要修改 v8-final 主代码；所有新实验使用独立版本目录和独立 models/reports/image 输出。'),
]: bullet(doc, text)

doc.add_heading(u('9. Skill 文档路径'), 1)
for text in [
    r'Fall detection / Crossbar 项目规范：C:\Users\lin\.codex\skills\fall-crossbar-project\SKILL.md',
    r'Word 文档生成与编辑规范：C:\Users\lin\.codex\plugins\cache\openai-primary-runtime\documents\26.727.11326\skills\documents\SKILL.md',
    r'Word 文档渲染验证脚本：C:\Users\lin\.codex\plugins\cache\openai-primary-runtime\documents\26.727.11326\skills\documents\render_docx.py',
]: bullet(doc, text)

doc.add_paragraph(u('结论：当前最可靠成果是 v8 p006 在 Subject4 上达到 78.38% Accuracy、78.24% Balanced Accuracy、80.00% ADL specificity 和 76.47% Fall recall。下一阶段核心是让 SNN fusion gate 保留跌倒相关的时序突变、姿态变化和落地后静止信息，同时避免压制主分支或误报 ADL。'))
doc.save(OUT)
print(OUT)
