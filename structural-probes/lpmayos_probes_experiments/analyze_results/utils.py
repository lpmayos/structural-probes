import json
import numpy as np
import plotly.graph_objs as go


def remove_empty_values(x, y):
    x = np.array(x)
    y = np.array(y)
    empty_indexes = np.where(y == None)[0]
    x = np.delete(x, empty_indexes)
    y = np.delete(y, empty_indexes)
    return x, y


def _fix_results(file_path):
    """
    To unify results, we keep 30 checkpoints
    """

    def _get_checkpoint_name(checkpoint, i):

        if 'output/checkpoint-' in checkpoint:
            new_checkpoint_name = int(checkpoint.split('/')[1].replace('checkpoint-', ''))
        elif 'output/model_state_epoch_' in checkpoint:
            new_checkpoint_name = i
        elif checkpoint == 'bert-base-cased':
            new_checkpoint_name = 0
        else:
            new_checkpoint_name = int(checkpoint)

        return new_checkpoint_name

    max_checkpoints = 30

    with open(file_path) as json_file:
        results = json.load(json_file)

        # 1. Let's assume that run1 is correct.

        # 1. remove checkpoints not probed from run1
        num_added = 0
        new_results = {}
        new_results['run1'] = {}
        for i, checkpoint in enumerate(results['run1']):
            new_checkpoint_name = _get_checkpoint_name(checkpoint, i)
            if results['run1'][checkpoint]['parse-depth']['dev.root_acc'] is not None:
                new_results['run1'][new_checkpoint_name] = results['run1'][checkpoint]
                num_added += 1
                if num_added == max_checkpoints:
                    break

        # 2. remove not wanted checkpoints from other runs

        for run in results:
            if run != 'run1':
                new_results[run] = {}
                for i, checkpoint in enumerate(results[run]):
                    new_checkpoint_name = _get_checkpoint_name(checkpoint, i)
                    if new_checkpoint_name in new_results['run1']:
                        new_results[run][new_checkpoint_name] = results[run][checkpoint]

        # 3. enforce same length (only 2 checkpoits are missing from SRL run5)

        for run in new_results:
            if new_results[run].keys() != new_results['run1'].keys():
                print('run %s of file_path %s is missing keys. We fix it by copying them from run1.' % (run, file_path))
                for checkpoint in new_results['run1']:
                    if checkpoint not in new_results[run]:
                        new_results[run][checkpoint] = new_results['run1'][checkpoint]

        # 4. enforce float numbers, not strings, for metrics
        for run in new_results:
            for checkpoint in new_results[run]:
                for metric in new_results[run][checkpoint]:
                    value = new_results[run][checkpoint][metric]
                    if isinstance(value, str) and '\n' in value:
                        new_results[run][checkpoint][metric] = float(value.replace('\n', ''))

        # 5. copy null values form other runs

        for run in new_results:
            for checkpoint in new_results[run]:
                if checkpoint != 0:
                    for metric in new_results[run][checkpoint]:

                        value = new_results[run][checkpoint][metric]
                        if value is None or (metric == 'parse-depth' and value['dev.root_acc'] is None) \
                                         or (metric == 'parse-distance' and value['dev.uuas'] is None):

                            copy_from = 'run1' if run != 'run1' else 'run2'
                            print('file_path %s; run %s; checkpoint %s; metric %s is null; we copy it from %s.' % (
                            file_path, run, checkpoint, metric, copy_from))
                            new_results[run][checkpoint][metric] = new_results[copy_from][checkpoint][metric]

        # 6. save new results

        new_file_path = file_path.replace('.json', '_fixed.json')
        with open(new_file_path, 'w') as outfile:
            json.dump(new_results, outfile, indent=4, sort_keys=True)


def load_traces(file_path, task, glue_task_name=None):
    with open(file_path) as json_file:
        results = json.load(json_file)
        traces = []
        for run in results:
            x_axis_values, traces_run = _get_run_data(results[run], run, task, glue_task_name)
            traces.extend(traces_run)

    return traces, x_axis_values


def _get_run_data(data, run_name, task, glue_task_name):
    if task == 'pos':
        return _get_pos_run_data(data, run_name)
    elif task == 'parsing':
        return _get_parsing_run_data(data, run_name)
    elif task == 'constituent_parsing':
        return _get_pap_constituents_run_data(data, run_name)
    elif task == 'squad':
        return _get_squad_run_data(data, run_name)
    elif task == 'mrpc':
        return _get_glue_run_data(data, run_name)
    elif task == 'qqp':
        return _get_glue_run_data(data, run_name)
    elif task == 'srl':
        return _get_srl_run_data(data, run_name)


def _get_squad_run_data(data, run_name):
    """ reads the data from a given run and creates a trace for each variable (squad_f1, depth_root_acc, etc)
    """

    depth_root_acc = []
    depth_spearmanr_mean = []
    distance_uuas = []
    distance_spearmanr_mean = []
    squad_exact_match = []
    squad_f1 = []
    mlm_perplexity = []

    for checkpoint in data:
        depth_root_acc.append(data[checkpoint]['parse-depth']['dev.root_acc'])
        depth_spearmanr_mean.append(data[checkpoint]['parse-depth']['dev.spearmanr-5_50-mean'])
        distance_uuas.append(data[checkpoint]['parse-distance']['dev.uuas'])
        distance_spearmanr_mean.append(data[checkpoint]['parse-distance']['dev.spearmanr-5_50-mean'])
        squad_exact_match.append(data[checkpoint]['squad_exact_match'])
        squad_f1.append(data[checkpoint]['squad_f1'])
        mlm_perplexity.append(data[checkpoint]['mlm_perplexity'])

    x_axis_values = [int(a) for a in data.keys()]

    x_clean, y_clean = remove_empty_values(x_axis_values, squad_f1)
    trace_squad_f1 = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='squad_f1_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, mlm_perplexity)
    trace_mlm_perplexity = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='mlm_perplexity_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_root_acc)
    trace_depth_root_acc = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_root_acc_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_uuas)
    trace_distance_uuas = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='distance_uuas_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_spearmanr_mean)
    trace_depth_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_spearmanr_mean_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_spearmanr_mean)
    trace_distance_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines',
                                               name='distance_spearmanr_mean_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, squad_exact_match)
    trace_squad_exact_match = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='squad_exact_match_' + run_name)

    traces = [trace_squad_f1, trace_mlm_perplexity, trace_depth_root_acc, trace_distance_uuas, trace_depth_spearmanr_mean, trace_distance_spearmanr_mean]
    # data = [trace_squad_f1, trace_mlm_perplexity, trace_depth_root_acc, trace_distance_uuas, trace_depth_spearmanr_mean, trace_distance_spearmanr_mean, trace_squad_exact_match]

    return x_axis_values, traces


def _get_pos_run_data(data, run_name):
    """ reads the data from a given run and creates a trace for each variable
    """

    depth_root_acc = []
    depth_spearmanr_mean = []
    distance_uuas = []
    distance_spearmanr_mean = []
    pos_acc = []
    pos_loss = []

    for checkpoint in data:
        depth_root_acc.append(data[checkpoint]['parse-depth']['dev.root_acc'])
        depth_spearmanr_mean.append(data[checkpoint]['parse-depth']['dev.spearmanr-5_50-mean'])
        distance_uuas.append(data[checkpoint]['parse-distance']['dev.uuas'])
        distance_spearmanr_mean.append(data[checkpoint]['parse-distance']['dev.spearmanr-5_50-mean'])
        pos_acc.append(data[checkpoint]['acc'])
        pos_loss.append(data[checkpoint]['loss'])

    x_axis_values = [int(a) for a in data.keys()]

    x_clean, y_clean = remove_empty_values(x_axis_values, pos_acc)
    trace_pos_acc = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='pos_acc_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, pos_loss)
    trace_pos_loss = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='pos_loss_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_root_acc)
    trace_depth_root_acc = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_root_acc_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_uuas)
    trace_distance_uuas = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='distance_uuas_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_spearmanr_mean)
    trace_depth_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_spearmanr_mean_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_spearmanr_mean)
    trace_distance_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines',
                                               name='distance_spearmanr_mean_' + run_name)

    traces = [trace_pos_acc, trace_pos_loss, trace_depth_root_acc, trace_distance_uuas, trace_depth_spearmanr_mean, trace_distance_spearmanr_mean]
    # TODO we could returns and plot all traces

    return x_axis_values, traces


def _get_srl_run_data(data, run_name):
    """ reads the data from a given run and creates a trace for each variable
    """

    depth_root_acc = []
    depth_spearmanr_mean = []
    distance_uuas = []
    distance_spearmanr_mean = []
    srl_f1_measure_overall = []
    srl_precision_overall = []
    srl_recall_overall = []
    srl_loss = []

    for checkpoint in data:
        depth_root_acc.append(data[checkpoint]['parse-depth']['dev.root_acc'])
        depth_spearmanr_mean.append(data[checkpoint]['parse-depth']['dev.spearmanr-5_50-mean'])
        distance_uuas.append(data[checkpoint]['parse-distance']['dev.uuas'])
        distance_spearmanr_mean.append(data[checkpoint]['parse-distance']['dev.spearmanr-5_50-mean'])
        srl_f1_measure_overall.append(data[checkpoint]['f1-measure-overall'])
        srl_precision_overall.append(data[checkpoint]['precision-overall'])
        srl_recall_overall.append(data[checkpoint]['recall-overall'])
        srl_loss.append(data[checkpoint]['loss'])

    x_axis_values = [a for a in range(len(data.keys()))]

    x_clean, y_clean = remove_empty_values(x_axis_values, srl_f1_measure_overall)
    trace_srl_f1_measure_overall = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='srl_f1_measure_overall_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, srl_precision_overall)
    trace_srl_precision_overall = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='srl_precision_overall_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, srl_recall_overall)
    trace_srl_recall_overall = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='srl_recall_overall_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, srl_loss)
    trace_srl_loss = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='srl_loss_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_root_acc)
    trace_depth_root_acc = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_root_acc_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_uuas)
    trace_distance_uuas = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='distance_uuas_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_spearmanr_mean)
    trace_depth_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_spearmanr_mean_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_spearmanr_mean)
    trace_distance_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines',
                                               name='distance_spearmanr_mean_' + run_name)

    traces = [trace_srl_f1_measure_overall, trace_srl_precision_overall, trace_srl_recall_overall, trace_srl_loss, trace_depth_root_acc, trace_distance_uuas, trace_depth_spearmanr_mean, trace_distance_spearmanr_mean]
    # TODO we could returns and plot all traces

    return x_axis_values, traces


def _get_pap_constituents_run_data(data, run_name):
    """ reads the data from a given run and creates a trace for each variable
    """

    depth_root_acc = []
    depth_spearmanr_mean = []
    distance_uuas = []
    distance_spearmanr_mean = []
    pap_acc = []
    pap_eval_score = []
    pap_loss = []

    for checkpoint in data:
        depth_root_acc.append(data[checkpoint]['parse-depth']['dev.root_acc'])
        depth_spearmanr_mean.append(data[checkpoint]['parse-depth']['dev.spearmanr-5_50-mean'])
        distance_uuas.append(data[checkpoint]['parse-distance']['dev.uuas'])
        distance_spearmanr_mean.append(data[checkpoint]['parse-distance']['dev.spearmanr-5_50-mean'])
        pap_acc.append(data[checkpoint]['acc'])
        pap_eval_score.append(data[checkpoint]['eval_score'])
        pap_loss.append(data[checkpoint]['loss'])

    x_axis_values = [a for a in range(len(data.keys()))]

    x_clean, y_clean = remove_empty_values(x_axis_values, pap_acc)
    trace_pap_acc = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='pap_acc_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, pap_eval_score)
    trace_pap_eval_score = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='pap_eval_score_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, pap_loss)
    trace_pap_loss = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='pap_loss_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_root_acc)
    trace_depth_root_acc = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_root_acc_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_uuas)
    trace_distance_uuas = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='distance_uuas_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_spearmanr_mean)
    trace_depth_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_spearmanr_mean_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_spearmanr_mean)
    trace_distance_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines',
                                               name='distance_spearmanr_mean_' + run_name)

    traces = [trace_pap_acc, trace_pap_eval_score, trace_pap_loss, trace_depth_root_acc, trace_distance_uuas, trace_depth_spearmanr_mean, trace_distance_spearmanr_mean]
    # TODO we could returns and plot all traces

    return x_axis_values, traces


def _get_parsing_run_data(data, run_name):
    """ reads the data from a given run and creates a trace for each variable
    """

    depth_root_acc = []
    depth_spearmanr_mean = []
    distance_uuas = []
    distance_spearmanr_mean = []
    parsing_uas = []
    parsing_las = []
    parsing_loss = []
    parsing_label_accuracy_score = []

    add_perplexity_trace = False
    mlm_perplexity = []

    for checkpoint in data:
        depth_root_acc.append(data[checkpoint]['parse-depth']['dev.root_acc'])
        depth_spearmanr_mean.append(data[checkpoint]['parse-depth']['dev.spearmanr-5_50-mean'])
        distance_uuas.append(data[checkpoint]['parse-distance']['dev.uuas'])
        distance_spearmanr_mean.append(data[checkpoint]['parse-distance']['dev.spearmanr-5_50-mean'])
        parsing_uas.append(data[checkpoint]['uas'])
        parsing_las.append(data[checkpoint]['las'])
        parsing_loss.append(data[checkpoint]['loss'])
        parsing_label_accuracy_score.append(data[checkpoint]['label accuracy score'])
        if 'mlm_perplexity' in data[checkpoint]:
            add_perplexity_trace = True
            mlm_perplexity.append(data[checkpoint]['mlm_perplexity'])

    x_axis_values = [int(a) for a in data.keys()]

    x_clean, y_clean = remove_empty_values(x_axis_values, parsing_uas)
    trace_parsing_uas = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='parsing_uas_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, parsing_las)
    trace_parsing_las = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='parsing_las_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, parsing_loss)
    trace_parsing_loss = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='parsing_loss_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, parsing_label_accuracy_score)
    trace_parsing_label_accuracy_score = go.Scatter(x=x_clean, y=y_clean, mode='lines',
                                                    name='parsing_label_accuracy_score_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_root_acc)
    trace_depth_root_acc = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_root_acc_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_uuas)
    trace_distance_uuas = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='distance_uuas_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_spearmanr_mean)
    trace_depth_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='depth_spearmanr_mean_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_spearmanr_mean)
    trace_distance_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean, mode='lines',
                                               name='distance_spearmanr_mean_' + run_name)

    if add_perplexity_trace:
        x_clean, y_clean = remove_empty_values(x_axis_values, mlm_perplexity)
        trace_mlm_perplexity = go.Scatter(x=x_clean, y=y_clean, mode='lines', name='mlm_perplexity_' + run_name)
        traces = [trace_parsing_uas, trace_parsing_las, trace_parsing_loss, trace_parsing_label_accuracy_score,
                  trace_depth_root_acc, trace_distance_uuas, trace_depth_spearmanr_mean, trace_distance_spearmanr_mean,
                  trace_mlm_perplexity]
    else:
        traces = [trace_parsing_uas, trace_parsing_las, trace_parsing_loss, trace_parsing_label_accuracy_score,
                  trace_depth_root_acc, trace_distance_uuas, trace_depth_spearmanr_mean, trace_distance_spearmanr_mean]
    # TODO we could returns and plot all traces

    return x_axis_values, traces


def _get_glue_run_data(data, run_name):
    """ reads the data from a given run and creates a trace for each variable (squad_f1, depth_root_acc, etc)
    """

    depth_root_acc = []
    depth_spearmanr_mean = []
    distance_uuas = []
    distance_spearmanr_mean = []
    glue_acc = []
    glue_f1 = []
    glue_acc_and_f1 = []

    for checkpoint in data:
        depth_root_acc.append(data[checkpoint]['parse-depth']['dev.root_acc'])
        depth_spearmanr_mean.append(data[checkpoint]['parse-depth']['dev.spearmanr-5_50-mean'])
        distance_uuas.append(data[checkpoint]['parse-distance']['dev.uuas'])
        distance_spearmanr_mean.append(data[checkpoint]['parse-distance']['dev.spearmanr-5_50-mean'])
        glue_acc.append(data[checkpoint]['acc'])
        glue_f1.append(data[checkpoint]['f1'])
        glue_acc_and_f1.append(data[checkpoint]['acc_and_f1'])

    x_axis_values = [int(a) for a in data.keys()]

    x_clean, y_clean = remove_empty_values(x_axis_values, glue_acc)
    trace_glue_acc = go.Scatter(x=x_clean, y=y_clean.astype(np.float), mode='lines', name='glue_acc_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, glue_f1)
    trace_glue_f1 = go.Scatter(x=x_clean, y=y_clean.astype(np.float), mode='lines', name='glue_f1_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, glue_acc_and_f1)
    trace_glue_acc_and_f1 = go.Scatter(x=x_clean, y=y_clean.astype(np.float), mode='lines', name='glue_acc_and_f1_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_root_acc)
    trace_depth_root_acc = go.Scatter(x=x_clean, y=y_clean.astype(np.float), mode='lines', name='depth_root_acc_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_uuas)
    trace_distance_uuas = go.Scatter(x=x_clean, y=y_clean.astype(np.float), mode='lines', name='distance_uuas_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, depth_spearmanr_mean)
    trace_depth_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean.astype(np.float), mode='lines', name='depth_spearmanr_mean_' + run_name)

    x_clean, y_clean = remove_empty_values(x_axis_values, distance_spearmanr_mean)
    trace_distance_spearmanr_mean = go.Scatter(x=x_clean, y=y_clean.astype(np.float), mode='lines', name='distance_spearmanr_mean_' + run_name)

    traces = [trace_glue_acc, trace_glue_f1, trace_glue_acc_and_f1, trace_depth_root_acc, trace_distance_uuas, trace_depth_spearmanr_mean, trace_distance_spearmanr_mean]
    # TODO we could returns and plot all traces

    return x_axis_values, traces


def _plot_figure(figure_name, traces, x_axis_values, x_axis_label, y_axis_label, y_axis_values=None, legend_dict=None, log_axis=False):

    x_axis_values = [a for a in range(30)]  # TODO we could pass this as a parameter

    fig = go.Figure(
        data=traces
    )

    show_legend = False if not legend_dict else True

    fig.layout.update(showlegend=show_legend,
                      margin=dict(r=0, l=0, b=0, t=0),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )

    if legend_dict:
        fig.update_layout(legend=legend_dict)

    fig.update_xaxes(
        ticktext=x_axis_values,
        tickvals=x_axis_values,
        title_text=x_axis_label,
        linecolor='LightGrey'
    )  # add gridcolor='LightGrey' to add inner lines

    if y_axis_values:
        fig.update_yaxes(
            range=y_axis_values,
        )

    fig.update_yaxes(
        title_text=y_axis_label,
        linecolor='LightGrey'
    )  # add gridcolor='LightGrey' to add inner lines

    if log_axis:
        fig.update_layout(yaxis_type="log")

    # requires: conda install -c plotly plotly-orca
    fig.write_image('/Users/lpmayos/Downloads/%s' % figure_name)

    fig.show()


def create_figure(data, traces_name, image_name, legend_dict, y_axis_label, y_axis_range, colors, markers, log_axis=False):
    x_axis_label = 'Fine-tuning checkpoints'

    data_to_plot = []
    for task_name, traces, x_axis_values in data:
        traces_to_plot = [a for a in traces if traces_name in a['name']]
        data_to_plot += _get_min_max_avg_traces(traces_to_plot, task_name, colors[task_name], markers[task_name])

    image_name = image_name

    _plot_figure(image_name, data_to_plot, x_axis_values, x_axis_label, y_axis_label, y_axis_range, legend_dict=legend_dict, log_axis=log_axis)


def create_figure_list(data, traces_name, y_axis_label, y_axis_range):
    x_axis_label = 'Fine-tuning checkpoints'

    for task_name, traces, x_axis_values in data:
        print(task_name)
        traces_to_plot = [a for a in traces if traces_name in a['name']]
        data_to_plot = _get_min_max_avg_traces(traces_to_plot, task_name)
        image_name = task_name.replace(' ', '_') + '_' + traces_name + '.png'
        _plot_figure(image_name, data_to_plot, x_axis_values, x_axis_label, y_axis_label, y_axis_range, None)


def plot_tasks_performance(data, all_possible_measures, log_axis=False):
    x_axis_label = 'Fine-tuning checkpoints'

    for task_name, traces, x_axis_values in data:
        for measure in all_possible_measures:
            data = [a for a in traces if measure in a['name']]
            if data:
                print('%s: %s' % (task_name, measure))
                data = _get_min_max_avg_traces(data, task_name)
                y_axis_label = measure
                image_name = '%s_%s.png' % (task_name.replace(' ', '_'), measure)

                _plot_figure(image_name, data, x_axis_values, x_axis_label, y_axis_label, None, None, log_axis)


def _get_min_max_avg_traces(data, task_name=None, task_color=None, task_marker=None):

    y1 = data[0]['y']
    y2 = data[1]['y']
    y3 = data[2]['y']
    y4 = data[3]['y']
    y5 = data[4]['y']

    data_stack = np.stack((y1, y2, y3, y4, y5), axis=0)

    upper_bound_data = np.max(data_stack, axis=0)
    lower_bound_data = np.min(data_stack, axis=0)
    avg_data = np.average(data_stack, axis=0)

    # x_data = data[0]['x']
    x_data = [a for a in range(31)]

    task_color = task_color if task_color else 'rgba(31, 119, 180, 1.0)'
    fill_color = task_color.replace('1.0', '0.1') if task_color else 'rgba(68, 68, 68, 0.1)'

    trace = go.Scatter(
        name=task_name if task_name else 'Average',
        x=x_data,
        y=avg_data,
        mode='lines+markers',
        marker_symbol=task_marker if task_marker else 'circle',
        marker_size=10,
        line=dict(color=task_color, width=2),
        fillcolor=fill_color,
        fill='tonexty')

    upper_bound = go.Scatter(
        name='Upper Bound',
        x=x_data,
        y=upper_bound_data,
        line=dict(color=task_color, width=0),
        fillcolor=fill_color,
        showlegend=False,
        fill='tonexty')

    lower_bound = go.Scatter(
        name='Lower Bound',
        x=x_data,
        y=lower_bound_data,
        line=dict(color=task_color, width=0),
        fillcolor=fill_color,
        showlegend=False)

    # Trace order can be important with continuous error bars
    traces = [lower_bound, trace, upper_bound]
    return traces


if __name__ == '__main__':

    # # Generate tractable results for all taks
    # _fix_results('bert_base_cased_finetuned_pos_results.json')
    # _fix_results('bert_base_cased_finetuned_squad_results.json')
    # _fix_results('bert_base_cased_finetuned_mrpc_results.json')
    # _fix_results('bert_base_cased_finetuned_qqp_results.json')
    # _fix_results('bert_base_cased_finetuned_parsing_results.json')
    # _fix_results('bert_base_cased_finetuned_parsing_multilingual_results.json')
    # _fix_results('bert_base_cased_finetuned_parsing_ptb_results.json')
    # _fix_results('bert_base_cased_finetuned_pap_constituents_results.json')
    # _fix_results('bert_base_cased_finetuned_srl_results.json')

    # _fix_results('bert_base_cased_finetuned_parsing_ptb_results.json')

    t_pos, x_pos = load_traces('bert_base_cased_finetuned_pos_results_fixed.json', 'pos')
    t_pars, x_pars = load_traces('bert_base_cased_finetuned_parsing_results_fixed.json', 'parsing')
    t_pars_mul, x_pars_mul = load_traces('bert_base_cased_finetuned_parsing_multilingual_results_fixed.json', 'parsing')
    t_pars_ptb, x_pars_ptb = load_traces('bert_base_cased_finetuned_parsing_ptb_results_fixed.json', 'parsing')
    t_pars_const, x_pars_const = load_traces('bert_base_cased_finetuned_pap_constituents_results_fixed.json',
                                             'constituent_parsing')
    t_squad, x_squad = load_traces('bert_base_cased_finetuned_squad_results_fixed.json', 'squad')
    t_squad_new, x_squad_new = load_traces('bert_base_cased_finetuned_squad_results_new.json', 'squad')
    t_qqpt, x_qqpt = load_traces('bert_base_cased_finetuned_qqp_results_fixed.json', 'qqp')
    t_mrpc, x_mrpc = load_traces('bert_base_cased_finetuned_mrpc_results_fixed.json', 'mrpc')
    t_srl, x_srl = load_traces('bert_base_cased_finetuned_srl_results_fixed.json', 'srl')

    all_data = [('PoS tagging', t_pos, x_pos),
                ('Dependency parsing; EN UD EWT', t_pars, x_pars),
                ('Dependency parsing; UD multilingual', t_pars_mul, x_pars_mul),
                ('Dependency parsing; PTB SD', t_pars_ptb, x_pars_ptb),
                ('Constituency parsing', t_pars_const, x_pars_const),
                ('Question answering', t_squad, x_squad),
                ('Paraphrase identification; QQPT', t_qqpt, x_qqpt),
                ('Paraphrase identification; MRPC', t_mrpc, x_mrpc),
                ('Semantic role labeling', t_srl, x_srl)]

    syntactic_data = [('PoS tagging', t_pos, x_pos),
                      ('Dependency parsing; EN UD EWT', t_pars, x_pars),
                      ('Dependency parsing; UD multilingual', t_pars_mul, x_pars_mul),
                      ('Dependency parsing; PTB SD', t_pars_ptb, x_pars_ptb),
                      ('Constituency parsing', t_pars_const, x_pars_const)]

    semantic_data = [('Question answering', t_squad, x_squad),
                     ('Paraphrase identification; QQPT', t_qqpt, x_qqpt),
                     ('Paraphrase identification; MRPC', t_mrpc, x_mrpc),
                     ('Semantic role labeling', t_srl, x_srl)]

    mlm_data = [('Dependency parsing; PTB SD', t_pars_ptb, x_pars_ptb),
                ('Question answering', t_squad_new, x_squad_new)]

    colors = {
        'PoS tagging': 'rgba(31, 119, 180, 1.0)',
        'Dependency parsing; EN UD EWT': 'rgba(255, 127, 14, 1.0)',
        'Dependency parsing; UD multilingual': 'rgba(214, 39, 40, 1.0)',
        'Dependency parsing; PTB SD': 'rgba(44, 160, 44, 1.0)',
        'Constituency parsing': 'rgba(148, 103, 189, 1.0)',
        'Question answering': 'rgba(255, 127, 14, 1.0)',
        'Paraphrase identification; QQPT': 'rgba(31, 119, 180, 1.0)',
        'Paraphrase identification; MRPC': 'rgba(44, 160, 44, 1.0)',
        'Semantic role labeling': 'rgba(238, 31, 90, 1.0)'}

    markers = {
        'PoS tagging': 'circle',
        'Dependency parsing; EN UD EWT': 'square',
        'Dependency parsing; UD multilingual': 'x',
        'Dependency parsing; PTB SD': 'diamond',
        'Constituency parsing': 'triangle-up',
        'Question answering': 'star',
        'Paraphrase identification; QQPT': 'hexagon2',
        'Paraphrase identification; MRPC': 'star-diamond',
        'Semantic role labeling': 'cross'}

    legend_outside_bottom = dict(
            orientation='h',
            y=-0.2,
            x=0.05,
            traceorder="normal",
            font=dict(family="sans-serif", size=14, color="black"),
            bgcolor="White",
            bordercolor="Black",
            borderwidth=1)

    legend_inside_bottom_right = dict(
            x=0.5,
            y=0.1,
            traceorder="normal",
            font=dict(family="sans-serif", size=14, color="black"),
            bgcolor="White",
            bordercolor="Black",
            borderwidth=1)

    legend_inside_middle_right = dict(
            x=0.5,
            y=0.45,
            traceorder="normal",
            font=dict(family="sans-serif", size=14, color="black"),
            bgcolor="White",
            bordercolor="Black",
            borderwidth=1)

    legend_inside_top_right = dict(
            x=0.60,
            y=0.90,
            traceorder="normal",
            font=dict(family="sans-serif", size=14, color="black"),
            bgcolor="White",
            bordercolor="Black",
            borderwidth=1)

    uuas_range = [0.5, 0.92]
    dspr_range = [0.5, 0.99]
    root_range = [0.5, 0.99]
    nspr_range = [0.5, 0.99]

    create_figure(mlm_data, 'mlm_perplexity', 'mlm_perplexity_evolution.png', legend_inside_top_right, 'Perplexity',
                  None, colors, markers)

