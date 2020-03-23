import json


print('MRPC ==========================================')
with open('/Users/lpmayos/code/structural-probes/structural-probes/lpmayos_probes_experiments/analyze_results/bert_base_cased_finetuned_glue_results.json') as f:
    data = json.load(f)
    for run in data:
        for task in data[run]:
            if task == 'MRPC':
                for checkpoint in data[run][task]:
                    if int(checkpoint) in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295, 305, 315, 325, 335, 345]:
                        path = ('/'.join([run, task, checkpoint]))
                        if data[run][task][checkpoint]['parse-distance']['dev.uuas'] == None:
                            print('Checkpoint %s missing parse-distance/dev.uuas' % path)

print('QQP ==========================================')
with open('/Users/lpmayos/code/structural-probes/structural-probes/lpmayos_probes_experiments/analyze_results/bert_base_cased_finetuned_glue_results.json') as f:
    data = json.load(f)
    for run in data:
        for task in data[run]:
            if task == 'QQP':
                for checkpoint in data[run][task]:
                    if int(checkpoint) in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000]:
                        path = ('/'.join([run, task, checkpoint]))
                        if data[run][task][checkpoint]['parse-distance']['dev.uuas'] == None:
                            print('Checkpoint %s missing parse-distance/dev.uuas' % path)

print('Parsing ==========================================')
with open('/Users/lpmayos/code/structural-probes/structural-probes/lpmayos_probes_experiments/analyze_results/bert_base_cased_finetuned_parsing_results.json') as f:
    data = json.load(f)
    for run in data:
        for checkpoint in data[run]:
            if int(checkpoint) in [20, 60, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 540, 580, 620, 660, 700, 740, 780, 820, 860, 900, 940, 980, 1020, 1060, 1100, 1140]:
                path = ('/'.join([run, checkpoint]))
                if data[run][checkpoint]['parse-distance']['dev.uuas'] == None:
                    print('Checkpoint %s missing parse-distance/dev.uuas' % path)

print('POS tagging ==========================================')
with open('/Users/lpmayos/code/structural-probes/structural-probes/lpmayos_probes_experiments/analyze_results/bert_base_cased_finetuned_pos_results.json') as f:
    data = json.load(f)
    for run in data:
        for checkpoint in data[run]:
            if int(checkpoint) in [20, 60, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 540, 580, 620, 660, 700, 740, 780, 820, 860, 900, 940, 980, 1020, 1060, 1100, 1140]:
                path = ('/'.join([run, checkpoint]))
                if data[run][checkpoint]['parse-distance']['dev.uuas'] == None:
                    print('Checkpoint %s missing parse-distance/dev.uuas' % path)

print('SQuAD ==========================================')
with open('/Users/lpmayos/code/structural-probes/structural-probes/lpmayos_probes_experiments/analyze_results/bert_base_cased_finetuned_squad_results.json') as f:
    data = json.load(f)
    for run in data:
        for checkpoint in data[run]:
            if int(checkpoint) in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000]:
                path = ('/'.join([run, checkpoint]))
                if data[run][checkpoint]['parse-distance']['dev.uuas'] == None:
                    print('Checkpoint %s missing parse-distance/dev.uuas' % path)