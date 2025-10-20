# Copyright 2025 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    This script is used to generate the stage-2 training data with LLMs for univariate time series, which can be further used as seed QAs for TSEvol.
    Usage:
        python3 -m chatts.sft.generate_uts_reason
"""

import numpy as np
import random
from tqdm import tqdm
import json
import yaml
from typing import *
from chatts.ts_generator.generate import generate_random_attributes, generate_time_series, generate_controlled_attributes, attribute_to_text, all_attribute_set
from chatts.utils.llm_utils import LLMClient, parse_llm_json
from chatts.utils.encoding_utils import timeseries_encoding, timeseries_to_list
from chatts.utils.attribute_utils import metric_to_controlled_attributes
from loguru import logger
import os


# CONFIG
TOTAL_CNT = yaml.safe_load(open("config/datagen_config.yaml"))['num_data_llm_qa']
SEQ_LEN = yaml.safe_load(open("config/datagen_config.yaml"))['seq_len']  # Set to None to enable random sequence length selection
ENCODING_METHOD = yaml.safe_load(open("config/datagen_config.yaml"))['encoding_method']
OUTPUT_BASE_DIR = yaml.safe_load(open("config/datagen_config.yaml"))['data_output_dir']
OUTPUT_DATASET = f'{OUTPUT_BASE_DIR}/llm_uts_reason_cn_{TOTAL_CNT}_{ENCODING_METHOD}.jsonl'
OUTPUT_LABEL = f'{OUTPUT_BASE_DIR}/evol_labels/llm_uts_reason_cn_{TOTAL_CNT}_{ENCODING_METHOD}.json'
DRYRUN = yaml.safe_load(open("config/datagen_config.yaml"))['dryrun']
LOCAL_LLM_PATH = yaml.safe_load(open("config/datagen_config.yaml"))['local_llm_path']
DISABLE_METRIC_CONFIG = yaml.safe_load(open("config/datagen_config.yaml"))["disable_metric_config"]
DISABLE_EXTREME_LENGTHS = yaml.safe_load(open("config/datagen_config.yaml"))["disable_extreme_lengths"]

# All Config for TS Attributes (type & probability)
metric_config = json.load(open('config/metric_set.json', 'rt'))

# 命名推理模板及示例问题
# 带占位符的命名推理模板
REASONING_TEMPLATES = {
    # --- 判断问题 (如果-那么) ---
    # 这些问题是自包含的，需要详细的解释。
    "多趋势异常判断": {
        "example": "如果异常定义为时间序列首先显示'上升'趋势，然后紧接着出现'下降'趋势，请分析这个时间序列。基于这个定义，从第0点到第256点的片段是否包含异常？",
        "question_type": "judgment"
    },
    # "趋势-局部交互判断": {
    #     "example": "'关键事件'定义为在整体'上升'趋势期间出现振幅大于30的'向上尖峰'。分析这个时间序列。基于这个规则，是否存在关键事件？",
    #     "question_type": "judgment"
    # },
    "连续局部特征判断": {
        "example": "如果'系统故障'定义为振幅>20的'突然增加'，在15个点内紧跟着振幅>15的'向下尖峰'，这个时间序列是否显示系统故障？",
        "question_type": "judgment"
    },
    # "稳定趋势扰动判断": {
    #     "example": "'失稳事件'定义为在长期'保持稳定'趋势中出现振幅为25或更大的'突然下降'。分析从第0点到第200点的时间序列。根据这个定义，它是否包含失稳事件？",
    #     "question_type": "judgment"
    # },
    # "多阶段趋势进展判断": {
    #     "example": "如果'正常增长周期'定义为按顺序的三个阶段：'上升'，然后'保持稳定'，然后再次'上升'，这个时间序列是否遵循正常增长周期模式？",
    #     "question_type": "judgment"
    # },
    # "矛盾信号判断": {
    #     "example": "如果'冲突信号'定义为在被识别为总体'上升'趋势的时期内观察到总振幅变化>40的'连续向下尖峰'，这个数据中是否存在冲突信号？",
    #     "question_type": "judgment"
    # },
    # "噪声阈值判断": {
    #     "example": "如果时间序列被分类为具有'噪声'环境（噪声标准差>0.2），而趋势被标记为'保持稳定'，噪声水平是否使稳定趋势分类失效？",
    #     "question_type": "judgment"
    # },
    # "季节性稳定性判断": {
    #     "example": "'稳定的季节性模式'定义为随时间变化振幅>1.0的'正弦周期性波动'。分析时间序列。季节性模式是否满足稳定性标准？",
    #     "question_type": "judgment"
    # },
    "统计判断": {
        "example": "如果任何低于10的数据点都被认为是'离群值'。这个时间序列是否应该被认为是异常的？",
        "question_type": "judgment"
    },
    "长期统计判断": {
        "example": "如果低于10且持续超过20个数据点被认为是异常的。这个时间序列是否应该被认为是异常的？",
        "question_type": "judgment"
    },
    "带恢复的长期统计判断": {
        "example": "如果数据点低于10且不恢复到原始状态被认为是异常的。时间序列中是否存在任何异常？",
        "question_type": "judgment"
    },
    # "实际应用中的多趋势异常判断": {
    #     "example": "对于一个新的移动应用，'失败的启动'定义为用户参与度最初增加但在第一周内开始下降。基于这个定义，应用在过去7天的用户参与度数据是否表明启动失败？",
    #     "question_type": "judgment"
    # },
    # "实际应用中的趋势-局部交互判断": {
    #     "example": "'关键服务器过载'定义为API响应时间'向上尖峰'超过800毫秒，发生在基线响应时间已显示渐进'上升'趋势的时期。分析服务器的性能数据。基于这个规则，是否存在关键服务器过载？",
    #     "question_type": "judgment"
    # },
    "实际应用中的连续局部特征判断": {
        "example": "应用程序中的'内存泄漏崩溃'通过其内存使用量'突然增加'超过200MB，然后在10分钟内出现超过150MB的急剧'向下尖峰'（表示崩溃和重启）来识别。这个应用程序的内存使用数据是否显示内存泄漏崩溃？",
        "question_type": "judgment"
    },
    "简单阈值判断": {
        "example": "低于10认为是异常的。判断这个时间序列里面有没有异常值？给出异常的区间",
        "question_type": "judgment"
    },
    "局部波动的简单阈值判断": {
        "example": "高于10的突刺认为是异常的。判断这个时间序列里面有没有异常值？",
        "question_type": "judgment"
    },
    "局部波动的最大/最小值判断": {
        "example": "如果出现了突降，并且降到了10以下，认为是异常的。判断这个时间序列里面有没有异常值？",
        "question_type": "judgment"
    },
    "振幅阈值异常判断": {
        "example": "如果时间序列中任何点的振幅变化超过5，就认为是异常波动。分析这个时间序列是否存在异常波动？",
        "question_type": "judgment"
    },
    "最大值阈值判断": {
        "example": "当时间序列的最大值超过50时，认为系统处于过载状态。根据数据判断系统是否过载？",
        "question_type": "judgment"
    },
    "最小值阈值判断": {
        "example": "如果时间序列的最小值低于5，认为系统运行异常。判断系统是否运行正常？",
        "question_type": "judgment"
    },
    "连续超阈值判断": {
        "example": "连续3个或以上数据点超过30的情况被认为是持续异常。这个时间序列中是否存在持续异常？",
        "question_type": "judgment"
    },
    "双阈值区间判断": {
        "example": "正常运行范围定义为15-25之间，超出此范围的数据点视为异常。分析时间序列是否有异常数据点？",
        "question_type": "judgment"
    },
    "变化率阈值判断": {
        "example": "相邻两点间的变化率超过20%被认为是剧烈波动。判断时间序列中是否存在剧烈波动？",
        "question_type": "judgment"
    },
    "峰值持续时间判断": {
        "example": "峰值（高于40）持续超过5个时间点被认为是异常峰值事件。时间序列中是否有异常峰值事件？",
        "question_type": "judgment"
    },
    "谷值深度判断": {
        "example": "谷值低于8且持续时间超过3个点被认为是深度下跌。判断是否存在深度下跌现象？",
        "question_type": "judgment"
    },
    "突增幅度判断": {
        "example": "单次突增幅度超过15的情况被认为是突发事件。时间序列中是否存在突发事件？",
        "question_type": "judgment"
    },
    "突降幅度判断": {
        "example": "单次突降幅度超过12的情况被认为是系统故障。判断是否存在系统故障？",
        "question_type": "judgment"
    },
    "平均值偏离判断": {
        "example": "偏离整体平均值超过2倍标准差的数据点被认为是离群值。时间序列中是否存在离群值？",
        "question_type": "judgment"
    },
    "累积超阈值判断": {
        "example": "累积超过阈值35的总时长超过10个时间点被认为是长期超负荷。判断是否存在长期超负荷？",
        "question_type": "judgment"
    },
    "多级阈值判断": {
        "example": "轻度异常（超过25）、中度异常（超过35）、重度异常（超过45）。根据这个分级，判断时间序列的异常等级？",
        "question_type": "judgment"
    },
    "阈值交叉频率判断": {
        "example": "在观察期内，数值穿越阈值线20超过5次被认为是不稳定状态。判断系统是否处于不稳定状态？",
        "question_type": "judgment"
    },
    "阈值恢复时间判断": {
        "example": "超过阈值28后，如果在3个时间点内未能恢复到阈值以下，认为是持续超标。是否存在持续超标情况？",
        "question_type": "judgment"
    },
    # "实际应用中的稳定趋势扰动判断": {
    #     "example": "电子商务产品的'供应链中断'定义为在销售本来保持稳定（'保持稳定'趋势）的时期内出现每小时销售量'突然下降'50件或更多。分析过去24小时的销售数据。根据这个定义，它是否包含供应链中断？",
    #     "question_type": "judgment"
    # },
    # "实际应用中的多阶段趋势进展判断": {
    #     "example": "新金融产品的'标准市场采用周期'定义为按顺序的三个阶段：交易量初期缓慢'上升'，随后是'保持稳定'的整固期，然后是另一个'上升'阶段。这个产品的交易量是否遵循标准市场采用周期？",
    #     "question_type": "judgment"
    # },
    # "实际应用中的矛盾信号判断": {
    #     "example": "对于工厂生产线，如果在本来显示生产总体'上升'的班次中观察到产量'连续向下尖峰'（总下降>40件/小时），则触发'机器健康警报'。这个数据中是否存在机器健康警报？",
    #     "question_type": "judgment"
    # },
    # "实际应用中的噪声阈值判断": {
    #     "example": "化学过程的物联网温度传感器应保持稳定。如果'不可靠数据'定义为由于环境噪声导致传感器读数标准差大于2°C，这种情况是否使底层过程温度稳定（'保持稳定'）的结论失效？",
    #     "question_type": "judgment"
    # },
    # "实际应用中的季节性稳定性判断": {
    #     "example": "电子商务网站的'稳定日常流量模式'定义为具有可预测的'正弦周期性波动'，其中高峰流量持续超过每小时1000用户。分析网站的流量数据。日常模式是否满足被认为稳定和重要的标准？",
    #     "question_type": "judgment"
    # },

    # --- 多选题 ---
    # 这些问题提供明确选项并需要证明。

    # "多趋势模式识别": {
    #     "example": "提供了服务器的CPU利用率数据。分析时间序列并确定：这种模式最能表明哪种情况？A）关键系统故障。B）服务器在重负载下达到处理能力极限。C）正常的日常周期。D）导致随机尖峰的软件错误。基于观察到的趋势进展解释您的选择。",
    #     "question_type": "multiple_choice"
    # },
    # "上下文中的局部特征解释": {
    #     "example": "提供了一个月内股票的价格数据。股票一直处于总体下跌趋势。如果您观察到任何重要的向上价格移动，最可能的解释是什么？A）股票下跌趋势的根本性逆转。B）简短的投机事件，可能由于新闻公告，对长期没有影响。C）季节性上涨的开始。D）数据报告错误。基于您的分析证明您的选择。",
    #     "question_type": "multiple_choice"
    # },
    # "组合特征的最佳匹配场景": {
    #     "example": "提供了凌晨2:00系统的网络流量数据。分析整体模式和任何异常事件。这种模式最能代表哪种情况？A）计划的数据迁移或系统备份。B）分布式拒绝服务（DDoS）攻击。C）正常的用户增长。D）网络硬件故障。基于您的分析提供推理。",
    #     "question_type": "multiple_choice"
    # },
    # "实际应用中的多趋势模式识别": {
    #     "example": "提供了几小时内服务器的CPU利用率数据。分析模式并确定：这种行为最能表明哪种情况？A）关键系统故障。B）服务器在重负载下达到处理能力极限。C）正常的日常周期。D）导致随机尖峰的软件错误。解释您的选择。",
    #     "question_type": "multiple_choice"
    # },
    # "实际应用中上下文的局部特征解释": {
    #     "example": "提供了一个月内股票的价格数据。分析时间序列中的任何重要事件或异常。如果您发现与总趋势相反的显著价格移动，最可能的解释是什么？A）趋势的根本性逆转。B）对长期无影响的简短投机事件。C）季节性模式的开始。D）数据报告错误。证明您的选择。",
    #     "question_type": "multiple_choice"
    # },
    # "实际应用中组合特征的最佳匹配场景": {
    #     "example": "提供了清晨时间系统的网络流量数据。分析模式并识别任何重要事件。这种模式最能代表哪种情况？A）计划的数据迁移或系统备份。B）分布式拒绝服务（DDoS）攻击。C）正常的用户增长。D）网络硬件故障。提供推理。",
    #     "question_type": "multiple_choice"
    # },

    # --- 开放式分析问题 ---

    # "多趋势影响分析": {
    #     "example": "提供了患者4小时内的血糖监测数据。分析模式并讨论这种行为对患者健康可能意味着什么。作为医疗保健提供者，您主要关心什么？",
    #     "question_type": "open_ended"
    # },
    # "趋势和局部特征综合": {
    #     "example": "提供了网络延迟监测数据。分析整体趋势和任何波动。任何观察到的变化如何影响您对网络稳定性的信心？解释您的分析思路。",
    #     "question_type": "open_ended"
    # },
    # "位置重要性分析": {
    #     "example": "提供了一天中电网的输出数据。分析任何重要事件及其时间。为什么任何重大事件的时间对您对电网稳定性的整体评估特别重要？",
    #     "question_type": "open_ended"
    # },
    # "实际应用中的多趋势影响分析": {
    #     "example": "提供了患者几小时内的血糖水平监测数据。分析模式并讨论这种行为对患者健康可能意味着什么。医疗保健提供者的主要关注点是什么？",
    #     "question_type": "open_ended"
    # },
    # "实际应用中的趋势和局部特征综合": {
    #     "example": "提供了随时间变化的网络延迟监测数据。分析整体模式和任何波动。这些观察如何影响您对网络稳定性的信心？解释您的分析思路。",
    #     "question_type": "open_ended"
    # },
    # "实际应用中的位置重要性分析": {
    #     "example": "提供了全天的电网输出数据，重点关注晚间时段。分析数据并解释为什么任何重要事件的时间对评估电网稳定性特别重要。",
    #     "question_type": "open_ended"
    # }
}


def generate_prompt_data():
    # Determine sequence length
    if SEQ_LEN is None:
        p = random.random()
        if p > 0.4:
            current_seq_len = 256
        elif p > 0.1 or DISABLE_EXTREME_LENGTHS:
            current_seq_len = random.randint(64, 1024)
        elif p > 0.05:
            current_seq_len = random.randint(5, 64)
        else:
            current_seq_len = random.randint(1024, 4096)
    else:
        current_seq_len = SEQ_LEN

    # Random choose a category and metric name
    sample = random.choice(list(metric_config))
    category = sample['category']
    metric = random.choice(sample['metrics'])
    fields = []

    # Choose a metric and generate
    # Generate attribute_pool and time series
    if DISABLE_METRIC_CONFIG:
        attribute_pool = generate_random_attributes(all_attribute_set['overall_attribute'], all_attribute_set['change'], seq_len=current_seq_len)
    else:
        attribute_pool = generate_controlled_attributes(metric_to_controlled_attributes(metric), seq_len=current_seq_len)

    attribute_pool['metric_name'] = metric
    attribute_pool['situation'] = category

    timeseries, attribute_pool = generate_time_series(attribute_pool, current_seq_len)

    # Scalar
    scaled_timeseries, cur_ts_prompt, _ = timeseries_encoding(timeseries, ENCODING_METHOD)

    # Generate QA
    instruction = f"这是一个名为{metric}的指标，来自{category}，长度为{current_seq_len}：{cur_ts_prompt}。"
    prompts = []
    fields = []

    # Generate random task
    task_candidates = ['reason']
    tasks = list(np.random.choice(task_candidates, size=1, replace=False))
    
    for task in tasks:
        prompt = f"我正在为时间序列分析大语言模型创建数据集。基于我提供的时间序列信息，我需要您根据指定的任务要求生成尽可能多的丰富问答对。这将用作大语言模型的训练数据。现在，我有一个来自{category}领域的名为{metric}的时间序列。"

        if task == 'reason':
            fields.append({'trend': [0], 'seasonal': [0], 'noise': [0], 'local': [0], 'statistic': [0]})
            prompt += "给定时间序列的特征如下："
            prompt += attribute_to_text(
                timeseries,
                attribute_pool,
                include_attributes=['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic'],
                generate_values=False
            )
            
            # Randomly select a named reasoning template.
            template_name = random.choice(list(REASONING_TEMPLATES.keys()))
            template_info = REASONING_TEMPLATES[template_name]
            
            # The example is a complete, self-contained, high-quality real-world question.
            example_question = template_info["example"]

            # 这是新部分：高质量问答对示例。
            example_answer_prompt = """
---
**示例答案（针对类似问题）：**

**答案示例（是）：**
观察时间序列数据，我可以识别两个关键组成部分：首先，基线响应时间在整个观察期内显示从约200毫秒到350毫秒的渐进上升趋势，满足'上升'趋势要求。其次，在第210个位置有一个达到950毫秒的突出尖峰，超过了定义中指定的800毫秒阈值。由于两个条件都满足——基线趋势上升且尖峰超过800毫秒。因此，答案应该是：是的，根据给定定义存在关键服务器过载。

**答案示例（否）：**
观察时间序列数据，我可以识别两个关键组成部分：首先，基线响应时间在整个观察期内保持在250-280毫秒相对稳定，显示'保持稳定'趋势而不是所需的'上升'趋势。其次，虽然在第180个位置有一个达到920毫秒的显著尖峰，但这个尖峰发生在稳定基线期间，而不是在上升趋势期间。由于第一个条件未满足——基线趋势没有按定义要求上升。因此，答案应该是：否，根据给定定义不存在关键服务器过载。

**答案示例（开放式）：**

观察时间序列数据，服务器表现出令人担忧的性能下降模式。基线响应时间显示从约200毫秒到350毫秒的渐进上升趋势，表明系统承受日益增加的压力或资源约束。更关键的是，在第210个位置有一个达到950毫秒的严重尖峰，这代表了基线的近3倍增长。这种模式表明服务器正在接近其容量极限并经历间歇性过载情况。我主要关心的是用户体验下降和潜在的系统不稳定性。我建议立即采取行动，包括：监控资源利用率以识别瓶颈，实施负载均衡或扩展解决方案，以及设置响应时间超过500毫秒的警报以防止未来事故。\n\n"""

            prompt += f"生成关于推理概念的问答对：**{template_name}**。这里是一个高质量的示例问题：'{example_question}'"
            prompt += example_answer_prompt
            prompt += f"""
现在，创建关于给定时间序列的新的、多样化的推理问题。问题必须是自包含的，设置在现实场景中，并为用户做出判断提供所有必要的定义。

**关键要求：**
1. **问题多样性**：创建具有不同问题格式和表达方式的混合问题。有些是简单问题，有些问题格式不规范。例如，有些问题可能只使用简单词汇，对答案格式没有要求，有些问题应该尽可能详细，对答案提出很多要求。输出列表必须包括**不同问题格式和类型**的问答对。
2. **自包含且现实**：确保每个问题都设置在真实世界背景下（如IT、金融、电子商务），并清楚定义判断标准（什么构成"异常"、"过载"等）。对于多选题，提供反映现实场景的**明确选项**。对于开放式问题，确保它们需要深入分析和推理。
3. **关注核心场景**：问题应主要探索整体趋势（包括多阶段趋势）和局部特征（如尖峰、下降或突然变化）之间的交互。
4. **要求深度推理**：答案必须遵循上述示例答案中显示的结构化格式：**分析** -> **证据** -> **推理** -> **结论**。它们必须解释结论背后的"为什么"，而不仅仅是陈述结论。答案应该丰富详细，提供对时间序列数据的全面分析。
5. **不同答案的问答对：** 对于是/否问题和多选题，您应该生成具有不同答案的问题和答案，如"是"、"否"、"A"、"B"等。对于开放式问题，您应该生成具有不同观点和结论的问题和答案。对于是/否问题，确保您清楚地得出答案并在推理部分末尾明确说明"是"或"否"。对于多选题，确保答案是选项之一（如"A"、"B"、"C"、"D"），并在推理部分末尾清楚说明答案。

**关键：最大化格式多样性**
创建多样化的问题和答案格式：

**问题风格示例：**
- "作为网络安全工程师监控过去6小时的流量数据。公司将'协调攻击'定义为持续15分钟以上、基线以上300的持续流量增长，结合连接超时尖峰。基于这个定义和观察到的数据模式，系统当前是否受到协调攻击？"
- "您正在为一个遇到结账延迟客户投诉的电子商务平台提供咨询。该平台将'关键性能下降'定义为：(1) 响应时间连续5分钟以上超过2秒，且 (2) 基线显示恶化趋势的任何时期。您对当前系统健康状况的专业评估是什么？"

**答案格式指南**（使用自然语言，不包括这些标签）：
答案应该**尽可能详细和长**，遵循这种结构：
[数据的初始观察] → [模式的具体证据] → [技术推理] → [明确结论]

混合自信/不确定语调、定量/定性推理、技术/商业语言、简洁/全面回应。回应应该非常丰富详细，提供对时间序列数据的全面分析。

注意您应该将初始问题和答案**重写**为**具有不同格式、词序和表达的多个问题和答案**！对于是/否问题，确保您清楚地得出答案并在推理部分末尾明确说明"是"或"否"。对于多选题，确保答案是选项之一（如"A"、"B"、"C"、"D"），并在推理部分末尾清楚说明答案。

**注意：** 如果给定的时间序列无法生成目标问答对，只需返回空列表。\n\n"""
        else:
            raise ValueError(f"Unknown task: {task}")

        prompt += """现在，请严格按照上述要求生成尽可能多的问答对（如果可以），并包含答案的参考文本。以JSON格式输出，例如：[{"question": "严格遵循任务问题1", "answer": "从数据中找到的答案1", "reference": "答案1的精确原始文本片段"}, {"question": "严格遵循任务问题2", "answer": "从数据中找到的答案2", "reference": "答案2的精确原始文本片段"}]。答案中包含的属性**必须从**给定的时间序列中找到，答案必须准确。生成的问答对不应重复，答案可以**非常详细**。在问题中**不得**提及具体的时间序列特征（例如，使用"振幅50的尖峰"、"时间序列中的突然增加"等词语），因为我们会提供这些信息。只使用"根据时间序列"或"根据第50点附近的值"等词语。"""

        prompts.append(prompt)

    # Generate final result
    result = []
    for prompt, field in zip(prompts, fields):
        result.append({
            'instruction': instruction,
            'prompt': prompt,
            'fields': field,
            'timeseries': [scaled_timeseries],
            'original_timeseries': [timeseries],
            'metrics': [metric],
            'attribute_pool': [attribute_pool],
            'corr_pool': []
        })

    return result

def check_answer_consistency(question: str, answer: str) -> str:
    """
    生成答案一致性检查的提示
    返回检查提示文本
    """
    consistency_prompt = f"""请仔细检查以下问答对中的答案是否存在逻辑矛盾或错误，特别注意数字比较错误。

问题：{question}

答案：{answer}

请检查答案中是否存在以下类型的错误：
1. **数字比较错误**：例如声称-13 > -10，或者5 < 3等明显错误的数字比较。注意结合语境仔细判断，发现其中的明显错误。
2. **结论与证据矛盾**：例如证据显示数值超过阈值，但结论说未超过阈值

如果答案中**没有发现**任何逻辑矛盾或明显错误，请回答"pass"。
如果答案中**存在**逻辑矛盾或明显错误，请回答"fail"。

只需回答"pass"或"fail"，不需要其他解释。"""
    
    return consistency_prompt

def generate_dataset():
    result = []
    prompts = []
    num_cnt = 0
    with tqdm(total=TOTAL_CNT, desc='生成提示中...') as t:
        cnt = 0
        while True:
            try:
                cur_data = generate_prompt_data()
            except ValueError as err:
                continue
            except IndexError as err:
                continue
            for item in cur_data:
                item['ts_idx'] = num_cnt
                result.append(item)
                prompts.append(item['prompt'])
                t.update()
                cnt += 1
            if cnt >= TOTAL_CNT:
                break
            num_cnt += 1

    # Use LLM to generate answer
    if DRYRUN:
        llm_answers = ['[{"question": "这是一个测试问题。", "answer": "这是一个测试答案。"}]'] * len(prompts)
        llm_client = None
    else:
        llm_client = LLMClient(model_path=LOCAL_LLM_PATH, engine='vllm', batch_size=8)
        llm_client.wait_for_ready()
        llm_answers = llm_client.llm_batch_generate(prompts, use_chat_template=True)
        llm_client.kill()

    # Parse json and collect all QA pairs for consistency check
    all_qa_pairs = []  # 存储所有解析出的QA对及其索引信息
    failed_cnt = 0
    
    for i in tqdm(range(len(result)), desc='解析中'):
        try:
            cur_qa_list = parse_llm_json(llm_answers[i])
            for j, qa in enumerate(cur_qa_list):
                all_qa_pairs.append({
                    'qa': qa,
                    'result_idx': i,
                    'qa_idx': j
                })
        except Exception as err:
            failed_cnt += 1
            continue
    
    # 批量进行一致性检查
    consistency_failed_cnt = 0
    valid_qa_pairs = []
    

    # Start a new llm_client
    if not DRYRUN and len(all_qa_pairs) > 0:
        llm_client = LLMClient(model_path="/mnt/bn/chatts/models/Qwen2.5-14B-Instruct", engine='vllm', batch_size=16)
        # 生成所有一致性检查的提示
        consistency_prompts = []
        for qa_pair in all_qa_pairs:
            consistency_prompt = check_answer_consistency(qa_pair['qa']['question'], qa_pair['qa']['answer'])
            consistency_prompts.append(consistency_prompt)
        
        # 批量调用LLM进行一致性检查
        try:
            consistency_results = llm_client.llm_batch_generate(consistency_prompts, use_chat_template=True)
            
            # 处理一致性检查结果
            for i, qa_pair in enumerate(all_qa_pairs):
                try:
                    result_text = consistency_results[i].strip().lower()
                    is_consistent = "pass" in result_text and "fail" not in result_text
                    
                    if is_consistent:
                        valid_qa_pairs.append(qa_pair)
                    else:
                        consistency_failed_cnt += 1
                        logger.warning(f"答案存在逻辑矛盾，已删除 - 问题: {qa_pair['qa']['question'][:100]}... 答案: {qa_pair['qa']['answer'][:200]}...")
                except Exception as e:
                    logger.warning(f"处理一致性检查结果时出错: {e}, 默认通过")
                    valid_qa_pairs.append(qa_pair)  # 出错时默认通过
                    
        except Exception as e:
            logger.warning(f"批量一致性检验过程中出现错误: {e}, 跳过一致性检查")
            valid_qa_pairs = all_qa_pairs  # 出错时使用所有QA对
    else:
        # DRYRUN模式或没有LLM客户端时，直接使用所有QA对
        valid_qa_pairs = all_qa_pairs

    # 构建最终的数据集
    dataset = []
    labels = []
    
    for qa_pair in valid_qa_pairs:
        qa = qa_pair['qa']
        result_idx = qa_pair['result_idx']
        
        dataset.append({
            'input': result[result_idx]['instruction'] + qa['question'],
            'output': qa['answer'],
            'timeseries': timeseries_to_list(result[result_idx]['timeseries'])
        })
        labels.append({
            'instruction': result[result_idx]['instruction'],
            'question': qa['question'],
            'fields': result[result_idx]['fields'],
            'ts_idx': result[result_idx]['ts_idx'],
            'metrics': result[result_idx]['metrics'],
            'corr_pool': result[result_idx]['corr_pool'],
            'attribute_pool': result[result_idx]['attribute_pool']
        })
    
    # 关闭LLM客户端
    if llm_client is not None:
        llm_client.kill()
        
    print(f"解析完成。解析失败次数：{failed_cnt}，一致性检验失败次数：{consistency_failed_cnt}，最终成功次数：{len(dataset)}。")

    return dataset, labels


if __name__ == '__main__':
    # Create output directory if not exists
    os.makedirs(os.path.dirname(OUTPUT_DATASET), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_LABEL), exist_ok=True)

    result, labels = generate_dataset()
    with open(OUTPUT_DATASET, 'wt') as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(OUTPUT_LABEL, 'wt') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"完成！已保存到 {OUTPUT_DATASET} 和 {OUTPUT_LABEL}。")
