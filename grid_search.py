import configparser
import os
import os.path

network = ['pairwise_lstm']

train_set = [('timit_speakers_100_50w_50m_not_reynolds_cluster', 100)]
test_set = [('timit_speakers_40_clustering_vs_reynolds', 40)]

margin_arcface = []
margin_cosface = [0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]
margin_sphereface = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

scale = [30]

rerun = True


best_rules = [0,1,1,0]

save_files = [('grid_search_all.csv', 0, False),
              ('grid_search_best_MR.csv', 0, True),
              ('grid_search_best_ARI.csv', 2, True),
              ('grid_search_best_DER.csv', 3, True)]

header = 'dataset,network,num train speakers,num test speakers,margin arcface,margin cosface,margin sphereface,scale,netfile name,min MR,max ACP,max ARI,min DER,\n'
base_path = 'grid_search_results/'

os.makedirs(base_path, exist_ok=True)

def update_results(dataset, net, train, test, margins, scale):
    global best_rules, save_files, header, base_path

    result_raw = []
    with open('common/data/result.csv', 'r') as f:
        result_raw = f.readlines()
    if len(result_raw) == 0:
        print('results could not be read')
        sys.exit(1)
    prefix1 = '%s,%s,%d,%d,%7.5f,%7.5f,%7.5f,%05.1f,'%(dataset,net,train[1],test[1],margins[0],margins[1],margins[2],scale)
    prefix2 = ' , , , , , , , ,'
    result = []
    for line in result_raw:
        parts = line.split(',')
        if len(parts) > 1:
            result.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),line))
    for file in save_files:
        if not os.path.exists(base_path+file[0]):
            with open(base_path+file[0], 'w+') as f:
                f.write(header)
        sorted_result=sorted(result, key=lambda res: res[file[1]])
        if best_rules[file[1]] == 1:
            sorted_result=sorted(result, key=lambda res: res[file[1]], reverse=True)
        text = prefix1+sorted_result[0][4]
        if not file[2]:
            for line in sorted_result[1:]:
                text += prefix2 + line[4]
            text += prefix2+' , , , , ,\n'
        with open(base_path+file[0], 'a') as f:
            f.write(text)

def build_config(dataset, net, train, test, margins, scale, rerun):
    config = configparser.ConfigParser()
    config.read_file(open('configs/config.cfg'))

    config['train']['pickle'] = train[0]
    config['train']['n_speakers'] = str(train[1])
    config['train']['rerun'] = 'False'
    if rerun:
        config['train']['rerun'] = 'True'
    config['test']['test_pickle'] = test[0]
    config['angular_loss']['margin_arcface'] = '%7.5f'%margins[0]
    config['angular_loss']['margin_cosface'] = '%7.5f'%margins[1]
    config['angular_loss']['margin_sphereface'] = '%7.5f'%margins[2]
    config['angular_loss']['scale'] = '%05.1f'%scale

    path = '%s/train_%d_test_%d/%s/angular_loss/arc_%7.5f_cos_%7.5f_sphere_%7.5f_scale_%05.1f'%(dataset,train[1],test[1],net,margins[0],margins[1],margins[2],scale)
    config['train']['run_name'] = path
    with open('configs/grid_config.cfg', 'w+') as f:
        config.write(f)

def run_net(net, train, test, margins, scale):
    global rerun
    dataset = 'timit'
    if net == 'pairwise_lstm_vox2':
        dataset = 'vox2'
    build_config(dataset, net, train, test, margins, scale, rerun)
    suffix = ' -train -test'
    if rerun:
        suffix = ''
    command = 'python controller.py -n '+net+suffix+' -plot -config grid_config.cfg'
    os.system(command)
    update_results(dataset, net, train, test, margins, scale)
    os.remove('common/data/result.csv')

for n in network:
    for train in train_set:
        for test in test_set:
            for s in scale:
                for ma in margin_arcface:
                    m = [ma, 0, 1]
                    run_net(n, train, test, m, s)
                for mc in margin_cosface:
                    m = [0, mc, 1]
                    run_net(n, train, test, m, s)
                for ms in margin_sphereface:
                    m = [0, 0, ms]
                    run_net(n, train, test, m, s)
