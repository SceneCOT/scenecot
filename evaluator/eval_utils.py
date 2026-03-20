import matplotlib.pyplot as plt
from collections import defaultdict

import json
import os
# from openai import AzureOpenAI
import traceback

import openai

def calculate_average_iou(result_list):
    iou_sums = defaultdict(float)
    iou_counts = defaultdict(int)
    iou_25_sums = defaultdict(float)
    iou_50_sums = defaultdict(float)
    for item in result_list:
        label = item['label'] # Get the label from the dictionary
        iou = item["iou"]
       
        iou_sums[label] += iou
        iou_counts[label] += 1
        if iou > 0.25:
            iou_25_sums[label] += 1
        if iou > 0.50:
            iou_50_sums[label] += 1

    # Calculate average IoU for each label
    average_ious = {label: iou_sums[label] / iou_counts[label] for label in iou_sums}
    average_ious_25 = {label: iou_25_sums[label] / iou_counts[label] for label in iou_25_sums}
    average_ious_50 = {label: iou_50_sums[label] / iou_counts[label] for label in iou_50_sums}

    return average_ious, average_ious_25, average_ious_50

def plot_top_average_iou_and_counts(average_ious, source_refer_data, top_n=20, name="Average IoU"):
    # Sort the labels by average IoU in descending order and select top top_n
    sorted_items = sorted(average_ious.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, avg_ious = zip(*sorted_items)
    training_counts = defaultdict(int)
    all_item = 0
    for item in source_refer_data:
        label = item['object_name']
        training_counts[label] += 1
        all_item += 1
    counts = [training_counts[label] for label in labels]

    print(f'counts: {counts}')
    print(f'all item count in scanrefer: {all_item}')
   
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot average IoU
    ax1.barh(labels, avg_ious, color='skyblue', label=f'{name}')
    ax1.set_xlabel(f'{name}')
    ax1.set_title(f'Top {top_n} Objects by {name} and Object Count')
    ax1.invert_yaxis()  # To display the highest IoU at the top

    # Create a secondary y-axis to plot the object counts
    ax2 = ax1.twiny()
    ax2.barh(labels, counts, color='orange', alpha=0.5, label='Object Count')
    ax2.set_xlabel('Object Count')

    # Add legends
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')

    plt.show()

def load_json(path):
    with open(path, 'r') as f:
        json_file = json.load(f)
    return json_file

def save_to_json(path, json_file):
    with open(path, 'w') as f:
        json.dump(json_file, f)

# def execute_chat(messages, api_key, model="gpt-3.5-turbo", region = "canadaeast", temperature=0.11, max_tokens=1024):
#     # REGION = "canadaeast"
#     # MODEL = "gpt-35-turbo-0125"
#     # API_KEY = "<your-key>"

#     API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
#     ENDPOINT = f"{API_BASE}/{region}"


#     client = AzureOpenAI(
#     api_key=api_key,
#     api_version="2024-02-01",
#     azure_endpoint=ENDPOINT,
#     )

#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#         )

#         # print(response.model_dump_json(indent=2))
#         # print(response.choices[0].message.content)
#         # time.sleep(0.5)
#         return response.choices[0].message.content if response.choices[0].message.content else ""
#     except Exception as e:
#         # print(e)
#         return ""

def execute_chat(
    messages: list,
    api_key: str = None,
    model: str = 'gpt-4o-2024-11-20',
):
    if api_key is not None:
        openai.api_key = api_key
    else:
        if 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        elif 'AZURE_OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
        else:
            raise LookupError("No OpenAI API key found")

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        raise e
    
def path_verify(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

import re

VICUNA_ACTION_TOKENS = {
    '给': 31999, '弘': 31998, '收': 31997, '왕': 31996, '黃': 31995, '还': 31994, '边': 31993, 'べ': 31992, 'げ': 31991, 'ὀ': 31990, '백': 31989, '泰': 31988, '역': 31987, '联': 31986, '怪': 31985, '奇': 31984, 'ɯ': 31983, '番': 31982, '止': 31981, '합': 31980, '才': 31979, 'ფ': 31978, '两': 31977, '명': 31976, '房': 31975, '候': 31974, '재': 31973, '교': 31972, '遠': 31971, '計': 31970, '故': 31969, '丁': 31968, 'ญ': 31967, '음': 31966, '進': 31965, 'ษ': 31964, '바': 31963, '모': 31962, '嘉': 31961, '双': 31960, '些': 31959, 'ヨ': 31958, 'ể': 31957, 'ഞ': 31956, '败': 31955, '茶': 31954, '회': 31953, '洲': 31952, '每': 31951, '월': 31950, '料': 31949, '梅': 31948, '深': 31947, 'ḏ': 31946, '방': 31945, '效': 31944, '导': 31943, 'Ē': 31942, '중': 31941, '내': 31940, '舞': 31939, 'ほ': 31938, 'Ġ': 31937, '１': 31936, '微': 31935, 'ន': 31934, '瀬': 31933, '唐': 31932, '助': 31931, '종': 31930, 'ˇ': 31929, '現': 31928, 'थ': 31927, '𝓝': 31926, '타': 31925, '居': 31924, 'ᵉ': 31923, 'သ': 31922, 'ഷ': 31921, 'ċ': 31920, 'პ': 31919, 'ව': 31918, 'ම': 31917, '删': 31916, '客': 31915, '兴': 31914, 'ശ': 31913, '昭': 31912, '员': 31911, '仮': 31910, '̌': 31909, '反': 31908, 'ぐ': 31907, '과': 31906, '装': 31905, '操': 31904, '连': 31903, '米': 31902, '构': 31901, '书': 31900, '⥤': 31899, '彦': 31898, 'ḳ': 31897, 'ྱ': 31896, '식': 31895, '运': 31894, '种': 31893, 'ҡ': 31892, '̍': 31891, 'ɵ': 31890, 'ദ': 31889, '项': 31888, '貴': 31887, '洞': 31886, '巴': 31885, 'ѫ': 31884, '達': 31883, '么': 31882, '\\u202d': 31881, 'ً': 31880, '▓': 31879, '˚': 31878, '飛': 31877, '頭': 31876, '孝': 31875, 'ự': 31874, 'Έ': 31873, 'Ÿ': 31872, '論': 31871, 'Ħ': 31870, '红': 31869, '庄': 31868, '军': 31867, 'ὺ': 31866, 'ක': 31865, 'ো': 31864, '健': 31863, '陈': 31862, 'ರ': 31861, 'ھ': 31860, '速': 31859, '渡': 31858, 'ਿ': 31857, '터': 31856, '食': 31855, '菜': 31854, '池': 31853, '话': 31852, '测': 31851, '溪': 31850, 'ក': 31849, '拳': 31848, '雅': 31847, '麻': 31846, '鳥': 31845, '越': 31844, '甲': 31843, 'ỳ': 31842, '希': 31841, '❯': 31840, '望': 31839, '非': 31838, '∇': 31837, '索': 31836, '确': 31835, 'む': 31834, 'ந': 31833, 'ϊ': 31832, '塔': 31831, '近': 31830, '群': 31829, 'ც': 31828, 'Ξ': 31827, '만': 31826, '銀': 31825, '斯': 31824, '喜': 31823, '학': 31822, '़': 31821, '鬼': 31820, '样': 31819, '丸': 31818, '차': 31817, 'զ': 31816, '衛': 31815, '尔': 31814, '坂': 31813, '話': 31812, '看': 31811, '复': 31810, 'ற': 31809, 'എ': 31808, '్': 31807, 'ӏ': 31806, 'ŝ': 31805, '들': 31804, '右': 31803, 'ḷ': 31802, 'ြ': 31801, 'ܝ': 31800, 'Ě': 31799, '达': 31798, 'ữ': 31797, 'ณ': 31796, '编': 31795, 'ˠ': 31794, '˜': 31793, '劉': 31792, '判': 31791, 'պ': 31790, '개': 31789, '隆': 31788, '试': 31787, '変': 31786, '告': 31785, '云': 31784, 'Ţ': 31783, 'ぶ': 31782, '씨': 31781, '座': 31780, '➖': 31779, 'ᾶ': 31778, 'ѐ': 31777, '।': 31776, 'ပ': 31775, '강': 31774, '經': 31773, 'ₗ': 31772, '⊤': 31771, '設': 31770, 'Ἐ': 31769, '击': 31768, '串': 31767, '∷': 31766, '々': 31765, 'ɫ': 31764, '母': 31763, '幸': 31762, 'ず': 31761, 'ף': 31760, '朱': 31759, '店': 31758, '切': 31757, '专': 31756, 'ỹ': 31755, '남': 31754, '岩': 31753, 'ṯ': 31752, '该': 31751, '雲': 31750, '桥': 31749, 'ķ': 31748, '면': 31747, '단': 31746, '错': 31745, '忠': 31744, 'ʎ': 31743, 'Ė': 31742, '羅': 31741, '沢': 31740, '楽': 31739, '✿': 31738, '용': 31737, '박': 31736, '默': 31735, '안': 31734, '再': 31733, 'आ': 31732, '雪': 31731, '富': 31730, '业': 31729, '陳': 31728, '航': 31727, 'Ἰ': 31726, 'į': 31725, '위': 31724, 'ရ': 31723, '足': 31722, '勝': 31721, 'շ': 31720, '̈': 31719, 'ゼ': 31718, 'হ': 31717, '무': 31716, 'ள': 31715, '樹': 31714, '昌': 31713, 'ා': 31712, '結': 31711, '草': 31710, '竹': 31709, 'ស': 31708, '藏': 31707, 'ふ': 31706, 'ལ': 31705, '活': 31704, '守': 31703, '址': 31702, '秀': 31701, '库': 31700, '군': 31699, '親': 31698, '御': 31697, '奈': 31696, '持': 31695, '官': 31694, 'ზ': 31693, '連': 31692, 'ਸ': 31691, '⅓': 31690, '付': 31689, '首': 31688, ' 身': 31687, 'শ': 31686, '称': 31685, 'ね': 31684, '断': 31683, '赤': 31682, '✅': 31681, '현': 31680, '电': 31679, 'ै': 31678, '̩': 31677, '智': 31676, '统': 31675, '引': 31674, 'ℂ': 31673, 'Ḫ': 31672, 'ץ': 31671, 'ʑ': 31670, '节': 31669, 'ή': 31668, 'ख': 31667, '并': 31666, 'গ': 31665, '߬': 31664, 'Ս': 31663, 'ા': 31662, '別': 31661, '兵': 31660, '恋': 31659, '问': 31658, '発': 31657, '打': 31656, '局': 31655, '屋': 31654, '若': 31653, '漢': 31652, '左': 31651, '令': 31650, '门': 31649, '気': 31648, '宝': 31647, 'ൻ': 31646, 'ợ': 31645, 'ེ': 31644, 'མ': 31643, '紀': 31642, '必': 31641, '换': 31640, '说': 31639, 'ൽ': 31638, '泉': 31637, 'ර': 31636, '育': 31635, '￼': 31634, '介': 31633, '场': 31632, '尾': 31631, 'ẓ': 31630, '函': 31629, '⇔': 31628, '戸': 31627, '╣': 31626, 'ൾ': 31625, '管': 31624, '್': 31623, 'ご': 31622, 'ゆ': 31621, 'ụ': 31620, '影': 31619, '移': 31618, '控': 31617, '乐': 31616, '技': 31615, 'ན': 31614, '态': 31613, '宿': 31612, '共': 31611, '页': 31610, 'න': 31609, '；': 31608, '그': 31607, '関': 31606, '素': 31605, 'ਰ': 31604, '호': 31603, '葉': 31602, 'ུ': 31601, '省': 31600, '展': 31599, 'ἡ': 31598, 'ˆ': 31597, '题': 31596, 'ী': 31595, '从': 31594, '汉': 31593, '夢': 31592, '⍵': 31591, '按': 31590, '▇': 31589, '┃': 31588, '車': 31587, '∉': 31586, 'ർ': 31585, '头': 31584, '－': 31583, '민': 31582, '聖': 31581, '死': 31580, '思': 31579, '세': 31578, '康': 31577, '∆': 31576, 'Մ': 31575, '̱': 31574, '输': 31573, 'ے': 31572, '년': 31571, '因': 31570, '秋': 31569, '视': 31568, 'រ': 31567, '广': 31566, '算': 31565, '業': 31564, '천': 31563, '選': 31562, '區': 31561, 'တ': 31560, '段': 31559, '起': 31558, '只': 31557, 'ủ': 31556, '\\x9d': 31555, 'ց': 31554, '黒': 31553, '়': 31552, '像': 31551, '⊂': 31550, '師': 31549, '处': 31548, 'ธ': 31547, '隊': 31546, '送': 31545, 'ὑ': 31544, '拉': 31543, '显': 31542, '支': 31541, '機': 31540, '球': 31539, '添': 31538, 'জ': 31537, '진': 31536, '万': 31535, '洋': 31534, '유': 31533, '线': 31532, '状': 31531, '马': 31530, '波': 31529, 'ℚ': 31528, '요': 31527, '载': 31526, '実': 31525, 'ユ': 31524, '‖': 31523, '想': 31522, 'Ď': 31521, '服': 31520, '報': 31519, 'ǧ': 31518, '를': 31517, '然': 31516, 'ⴰ': 31515, 'ἱ': 31514, 'ɹ': 31513, '\\x99': 31512, '☉': 31511, '克': 31510, '鉄': 31509, 'Ṭ': 31508, '例': 31507, '老': 31506, '语': 31505, '張': 31504, '宇': 31503, '何': 31502, 'ペ': 31501, '̂': 31500, 'ⁿ': 31499, 'ိ': 31498, 'ք': 31497, '湖': 31496, '景': 31495, '🌍': 31494, '드': 31493, '∙': 31492, '黄': 31491, 'ǫ': 31490, 'Ḩ': 31489, 'հ': 31488, '비': 31487, '⊗': 31486, 'ි': 31485, '森': 31484, '┈': 31483, '今': 31482, 'ய': 31481, '超': 31480, '写': 31479, '【': 31478, '⸮': 31477, '沙': 31476, '去': 31475, '意': 31474, '包': 31473, '】': 31472, '传': 31471, 'ʋ': 31470, 'ύ': 31469, 'Ă': 31468, '曲': 31467, '计': 31466, '∣': 31465, '♀': 31464, '序': 31463, '变': 31462, '密': 31461, '◦': 31460, 'န': 31459, '산': 31458, '여': 31457, '帝': 31456, '究': 31455, '布': 31454, '็': 31453, 'ི': 31452, '登': 31451, '任': 31450, '港': 31449, 'ホ': 31448, 'ड': 31447, '岡': 31446, '伝': 31445, 'ḩ': 31444, 'ղ': 31443, '編': 31442, '创': 31441, '\\x91': 31440, '认': 31439, '術': 31438, 'ध': 31437, '及': 31436, '해': 31435, 'բ': 31434, '站': 31433, '角': 31432, 'ĉ': 31431, '阳': 31430, '机': 31429, 'ை': 31428, '商': 31427, 'Ά': 31426, '七': 31425, '现': 31424, '没': 31423, 'ื': 31422, 'ܐ': 31421, '造': 31420, '比': 31419, '⌘': 31418, '마': 31417, '崎': 31416, '转': 31415, 'ょ': 31414, 'ू': 31413, '经': 31412, '會': 31411, '记': 31410, '株': 31409, '조': 31408, '被': 31407, '문': 31406, 'Ζ': 31405, '開': 31404, '则': 31403, 'ォ': 31402, 'ང': 31401, '良': 31400, '品': 31399, '交': 31398, 'ṅ': 31397, 'ู': 31396, '玉': 31395, 'Ī': 31394, '根': 31393, '橋': 31392, '或': 31391, '夜': 31390, '此': 31389, 'へ': 31388, 'դ': 31387, 'প': 31386, '電': 31385, 'ச': 31384, '需': 31383, '模': 31382, '们': 31381, 'भ': 31380, '\\u202c': 31379, '경': 31378, 'ण': 31377, '求': 31376, 'Ψ': 31375, '章': 31374, '友': 31373, '╚': 31372, 'က': 31371, '应': 31370, '失': 31369, '注': 31368, '研': 31367, '完': 31366, '津': 31365, 'โ': 31364, '軍': 31363, '미': 31362, '配': 31361, '属': 31360, '基': 31359, '务': 31358, '線': 31357, '那': 31356, 'ʷ': 31355, '은': 31354, '\\u2028': 31353, '无': 31352, '╔': 31351, 'अ': 31350, '义': 31349, '\\x9c': 31348, '久': 31347, '오': 31346, '선': 31345, 'ད': 31344, 'ề': 31343, 'അ': 31342, 'ἔ': 31341, 'ု': 31340, 'ך': 31339, '堂': 31338, '仁': 31337, 'ʐ': 31336, 'ゲ': 31335, '공': 31334, '选': 31333, 'ῥ': 31332, '向': 31331, 'ष': 31330, 'ट': 31329, '张': 31328, '우': 31327, 'བ': 31326, '而': 31325, 'ា': 31324, 'թ': 31323, '雄': 31322, '九': 31321, '结': 31320, '□': 31319, 'ứ': 31318, '̪': 31317, '⊥': 31316, '佐': 31315, 'Ṣ': 31314, '火': 31313, 'ゃ': 31312, 'Ű': 31311, 'ข': 31310, 'ϵ': 31309, '伊': 31308, 'Հ': 31307, '제': 31306, '形': 31305, '六': 31304, 'ĝ': 31303, '提': 31302, '්': 31301, '龙': 31300, '장': 31299, 'び': 31298, 'ᴇ': 31297, '宗': 31296, '未': 31295, '容': 31294, '국': 31293, 'င': 31292, '陽': 31291, '已': 31290, '┤': 31289, '영': 31288, 'ひ': 31287, '을': 31286, '연': 31285, 'ള': 31284, '录': 31283, '▲': 31282, '‾': 31281, 'ớ': 31280, '부': 31279, 'ʌ': 31278, '符': 31277, '消': 31276, '♣': 31275, '學': 31274, '修': 31273, '由': 31272, 'ქ': 31271, 'ヴ': 31270, '╝': 31269, '调': 31268, '与': 31267, '华': 31266, 'ὲ': 31265, '改': 31264, '组': 31263, '신': 31262, '̄': 31261, '府': 31260, '典': 31259, 'ヤ': 31258, 'ἄ': 31257, 'գ': 31256, 'ギ': 31255, 'ば': 31254, 'ன': 31253, 'ไ': 31252, 'ヒ': 31251, 'ど': 31250, 'வ': 31249, 'ਾ': 31248, 'ძ': 31247, 'შ': 31246, '➜': 31245, '先': 31244, '言': 31243, '\\x81': 31242, '夏': 31241, '君': 31240, '龍': 31239, '就': 31238, '命': 31237, '○': 31236, 'լ': 31235, '▸': 31234, 'မ': 31233, 'ར': 31232, '구': 31231, '∫': 31230, '户': 31229, 'ေ': 31228, '阿': 31227, 'ە': 31226, '화': 31225, '≃': 31224, 'ல': 31223, '网': 31222, '他': 31221, '後': 31220, 'ὁ': 31219, 'য': 31218, '条': 31217, '╩': 31216, '╗': 31215, '̣': 31214, '查': 31213, 'ұ': 31212, '̥': 31211, 'Û': 31210, '無': 31209, 'ག': 31208, '나': 31207, 'ろ': 31206, 'ポ': 31205, 'দ': 31204, '男': 31203, '〜': 31202, '解': 31201, '⊕': 31200, '보': 31199, '원': 31198, '라': 31197, '博': 31196, '实': 31195, 'ׁ': 31194, '源': 31193, '見': 31192, '否': 31191, '常': 31190, '소': 31189, '↵': 31188, '華': 31187, '∼': 31186, '系': 31185, '等': 31184, '码': 31183, '放': 31182, '土': 31181, '量': 31180, ' 園': 31179, '⊢': 31178, '트': 31177, '夫': 31176, '限': 31175, '进': 31174, '歌': 31173, 'ピ': 31172, '☺': 31171, '전': 31170, '德': 31169, '格': 31168, 'ʀ': 31167, '单': 31166, 'ɣ': 31165, 'ட': 31164, '朝': 31163, 'Ť': 31162, '館': 31161, 'ắ': 31160, '千': 31159, '상': 31158, '直': 31157, '永': 31156, '្': 31155, 'ু': 31154, '일': 31153, '除': 31152, '流': 31151, 'ত': 31150, '其': 31149, 'স': 31148, 'Ъ': 31147, 'ണ': 31146, 'ấ': 31145, '英': 31144, '长': 31143, 'ậ': 31142, '特': 31141, '皇': 31140, 'վ': 31139, '过': 31138, '고': 31137, '도': 31136, '♂': 31135, ' 功': 31134, '象': 31133, 'च': 31132, '義': 31131, 'ხ': 31130, '어': 31129, '╦': 31128, 'Ə': 31127, '성': 31126, '参': 31125, '動': 31124, 'ザ': 31123, '片': 31122, '福': 31121, '初': 31120, '┘': 31119, '∅': 31118, '期': 31117, '،': 31116, 'じ': 31115, '♯': 31114, '香': 31113, '谷': 31112, 'や': 31111, 'そ': 31110, '周': 31109, '県': 31108, '利': 31107, 'ച': 31106, 'ũ': 31105, 'ོ': 31104, '郡': 31103, '김': 31102, '程': 31101, '更': 31100, 'ң': 31099, '魔': 31098, '̲': 31097, '志': 31096, 'せ': 31095, '↳': 31094, '서': 31093, '接': 31092, 'ό': 31091, '風': 31090, '≫': 31089, '请': 31088, '馬': 31087, '返': 31086, '色': 31085, '指': 31084, '∗': 31083, '┐': 31082, '는': 31081, 'ֶ': 31080, 'ℓ': 31079, 'Ù': 31078, 'ғ': 31077, '好': 31076, '門': 31075, ' 力': 31074, 'แ': 31073, '制': 31072, '校': 31071, 'ภ': 31070, '間': 31069, 'わ': 31068, '♠': 31067, '外': 31066, 'ֵ': 31065, 'ὴ': 31064, '니': 31063, '标': 31062, 'ベ': 31061, '∑': 31060, 'έ': 31059, 'ġ': 31058, '关': 31057, 'ṛ': 31056, 'ল': 31055, '에': 31054, 'ာ': 31053, '氏': 31052, 'ソ': 31051, '得': 31050, '記': 31049, '☆': 31048, '百': 31047, '画': 31046, '場': 31045, ' 八': 31044, '知': 31043, 'ά': 31042, '工': 31041, 'ĩ': 31040, 'း': 31039, 'ネ': 31038, '台': 31037, 'ɒ': 31036, 'ศ': 31035, 'ས': 31034, '吉': 31033, '治': 31032, '春': 31031, '科': 31030, 'კ': 31029, 'ワ': 31028, 'ტ': 31027, '开': 31026, '列': 31025, '获': 31024, '教': 31023, '少': 31022, '息': 31021, '始': 31020, 'ṃ': 31019, '松': 31018, 'ﬁ': 31017, '间': 31016, 'ா': 31015, '政': 31014, '자': 31013, 'ब': 31012, 'Ա': 31011, 'ป': 31010, 'श': 31009, 'ļ': 31008, '『': 31007, 'ম': 31006, '』': 31005, '宮': 31004, 'ボ': 31003, '┌': 31002, 'Υ': 31001, '동': 31000
}

COT_INDICATORS = {
    "think_type": ["<think_type>", "</think_type>"],
    "think_grd": ["<think_grd>", "</think_grd>"],
    "think_rgn": ["<think_rgn>", "</think_rgn>"],
    "OBJ": ["[OBJ]"],
    "think_task": ["<think_task>", "</think_task>"],
    "list_obj_prob": ["<list_obj_prob>"],
    "list_obj_loc_prob": ["<list_obj_loc_prob>"],
    "list_rgn_obj": ["<list_rgn_obj>"],
    "highlight_obj": ["<highlight_obj>"],
    "img_token_indicator": ["<img_start>", "<img_end>"],
    "obj_prob": ["<obj_prob>", "</obj_prob>"],
    "obj_cap": ["<obj_cap>", "</obj_cap>"],
    "obj_loc_prob": ["<obj_loc_prob>", "</obj_loc_prob>"],
    "obj_loc_plr_prob": ["<obj_loc_plr_prob>", "</obj_loc_plr_prob>"],
    "list_obj_loc_plr_prob": ["<list_obj_loc_plr_prob>"],
    "think_sum": ["<think_sum>", "</think_sum>"],
    "answer": ["<answer>", "</answer>"],
}

COT_INDICATORS_LIST = []
for k,v in COT_INDICATORS.items():
    COT_INDICATORS_LIST.extend(v)

COT_INDICATORS_TOKENIZE = {
    k: v for k,v in zip(COT_INDICATORS_LIST, list(VICUNA_ACTION_TOKENS.keys())[-len(COT_INDICATORS_LIST):])   
}

COT_INDICATORS_DETOKENIZE = {
    v: k for k,v in zip(COT_INDICATORS_LIST, list(VICUNA_ACTION_TOKENS.keys())[-len(COT_INDICATORS_LIST):])   
}

GRD_TOKEN_TXT = COT_INDICATORS_TOKENIZE[COT_INDICATORS['OBJ'][0]]
OBJ_PROB_START_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<obj_prob>']]
OBJ_PROB_END_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['</obj_prob>']]
OBJ_LOC_START_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<obj_loc_prob>']]
OBJ_LOC_END_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['</obj_loc_prob>']]
OBJ_LOC_PLR_START_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<obj_loc_plr_prob>']]
OBJ_LOC_PLR_END_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['</obj_loc_plr_prob>']]
LIST_OBJ_PROB_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<list_obj_prob>']]
LIST_OBJ_LOC_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<list_obj_loc_prob>']]
LIST_OBJ_LOC_PLR_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<list_obj_loc_plr_prob>']]
LIST_RGN_OBJ_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<list_rgn_obj>']]
HIGHLIGHT_OBJ_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<highlight_obj>']]
IMG_START_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<img_start>']]
IMG_END_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<img_end>']]
GRD_TXT_LEFT_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['<think_grd>']]
GRD_TXT_RIGHT_TOKEN_ID = VICUNA_ACTION_TOKENS[COT_INDICATORS_TOKENIZE['</think_grd>']]

def parse_cot_answer(text):
    parsed_data = {}
    
    for key, markers in COT_INDICATORS.items():
        start_marker, end_marker = markers[0], markers[-1]
        pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            parsed_data[key] = match.group(1).strip()
        elif len(markers) == 1 and start_marker in text:  # Single-token indicators like "[OBJ]"
            parsed_data[key] = True  # Presence flag
        else:
            parsed_data[key] = "Empty content for this key!"  # fill with empty string if matching not found

    return parsed_data

def replace_cot_indicators_with_tokens(text):   # used in data/datasets.py
    for indicator, token in COT_INDICATORS_TOKENIZE.items():
        text = text.replace(indicator, token)
    return text

def replace_cot_tokens_with_indicators(text):   # used in model/leo_cot_agent.py
    for indicator, token in COT_INDICATORS_TOKENIZE.items():
        text = text.replace(token, indicator)
    return text