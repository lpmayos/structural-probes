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


def _fix_results(results):
    """
    Due to some random issues with the HPC, some runs got stuck and did not finish.
    We mock that missing data with data from other runs just to be able to plot results.
    """

    # remove checkpoints not probed
    new_results = {}
    for run in results:
        new_results[run] = {}
        for checkpoint in results[run]:
            if results[run][checkpoint]['parse-depth']['dev.root_acc'] is not None:
                new_results[run][checkpoint] = results[run][checkpoint]

    # downsize to 30 checkpoints
    for run in new_results:
        if len(new_results[run].keys()) != 30:
            print('babau')


    return new_results


def load_traces(file_path, task, glue_task_name=None):
    with open(file_path) as json_file:
        results = json.load(json_file)
        results = _fix_results(results)
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
    elif task == 'glue':
        return _get_glue_run_data(data[glue_task_name], run_name)
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

    new_data = {}
    for checkpoint in data:
        new_checkpoint = int(checkpoint.split('/')[1].replace('checkpoint-', ''))
        new_data[new_checkpoint] = data[checkpoint]
    data = new_data

    for checkpoint in sorted(data):
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

    for checkpoint in data:
        depth_root_acc.append(data[checkpoint]['parse-depth']['dev.root_acc'])
        depth_spearmanr_mean.append(data[checkpoint]['parse-depth']['dev.spearmanr-5_50-mean'])
        distance_uuas.append(data[checkpoint]['parse-distance']['dev.uuas'])
        distance_spearmanr_mean.append(data[checkpoint]['parse-distance']['dev.spearmanr-5_50-mean'])
        parsing_uas.append(data[checkpoint]['uas'])
        parsing_las.append(data[checkpoint]['las'])
        parsing_loss.append(data[checkpoint]['loss'])
        parsing_label_accuracy_score.append(data[checkpoint]['label accuracy score'])

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

    traces = [trace_parsing_uas, trace_parsing_las, trace_parsing_loss, trace_parsing_label_accuracy_score, trace_depth_root_acc, trace_distance_uuas, trace_depth_spearmanr_mean, trace_distance_spearmanr_mean]
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


def _plot_figure(figure_name, traces, x_axis_values, x_axis_label, y_axis_label, y_axis_values=None, legend_dict=None):
    fig = go.Figure(
        data=traces
    )

    show_legend = False if not legend_dict else True

    fig.layout.update(showlegend=show_legend, margin=dict(r=0, l=0, b=0, t=0))

    if legend_dict:
        fig.update_layout(legend=legend_dict)

    fig.update_xaxes(
        ticktext=x_axis_values,
        tickvals=x_axis_values,
        title_text=x_axis_label
    )

    if y_axis_values:
        fig.update_yaxes(
            range=y_axis_values,
        )

    fig.update_yaxes(title_text=y_axis_label)

    # requires: conda install -c plotly plotly-orca
    fig.write_image('/Users/lpmayos/Downloads/%s' % figure_name)

    fig.show()


def create_figure(data, traces_name, traces_style, image_name, legend_dict, y_axis_label, y_axis_range, colors, dash_types):
    x_axis_label = 'Finetuning checkpoints'

    data_to_plot = []
    for task_name, traces, x_axis_values in data:
        traces_to_plot = [a for a in traces if traces_name in a['name']]
        style = 'solid' if not traces_style else dash_types[task_name]
        data_to_plot += _get_min_max_avg_traces(traces_to_plot, task_name, style, colors[task_name])

    image_name = image_name

    _plot_figure(image_name, data_to_plot, x_axis_values, x_axis_label, y_axis_label, y_axis_range, legend_dict=legend_dict)


def create_figure_list(data, traces_name, y_axis_label, y_axis_range):
    x_axis_label = 'Finetuning checkpoints'

    for task_name, traces, x_axis_values in data:
        print(task_name)
        traces_to_plot = [a for a in traces if traces_name in a['name']]
        data_to_plot = _get_min_max_avg_traces(traces_to_plot, task_name, 'solid')
        image_name = task_name.replace(' ', '_') + '_' + traces_name + '.png'
        _plot_figure(image_name, data_to_plot, x_axis_values, x_axis_label, y_axis_label, y_axis_range, None)


def plot_tasks_performance(data, all_possible_measures):
    x_axis_label = 'Finetuning checkpoints'

    for task_name, traces, x_axis_values in data:
        for measure in all_possible_measures:
            data = [a for a in traces if measure in a['name']]
            if data:
                print('%s: %s' % (task_name, measure))
                data = _get_min_max_avg_traces(data, task_name, 'solid')
                y_axis_label = measure
                image_name = '%s_%s.png' % (task_name.replace(' ', '_'), measure)

                _plot_figure(image_name, data, x_axis_values, x_axis_label, y_axis_label, None, None)


def _get_min_max_avg_traces(data, task_name=None, dash_type='solid', task_color=None):

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

    upper_bound = go.Scatter(
        name='Upper Bound',
        x=x_data,
        y=upper_bound_data,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False)

    trace = go.Scatter(
        name=task_name if task_name else 'Average',
        x=x_data,
        y=avg_data,
        mode='lines',
        line=dict(color=task_color, dash=dash_type) if task_color else dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    lower_bound = go.Scatter(
        name='Lower Bound',
        x=x_data,
        y=lower_bound_data,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        showlegend=False)

    # Trace order can be important with continuous error bars
    traces = [lower_bound, trace, upper_bound]
    return traces


if __name__ == '__main__':

    file_path = 'bert_base_cased_finetuned_pos_results.json'
    with open(file_path) as json_file:
        results = json.load(json_file)
        new_results = _fix_results(results)
        with open('babau.json', 'w') as outfile:
            json.dump(new_results, outfile, indent=4, sort_keys=True)

