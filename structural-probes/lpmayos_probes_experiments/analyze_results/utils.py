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


def get_squad_run_data(data, run_name):
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


def get_pos_run_data(data, run_name):
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


def get_parsing_run_data(data, run_name):
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


def get_glue_run_data(data, run_name):
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


def plot_figure(traces, x_axis_values, x_axis_text, title, x_axis_label, y_axis_label, y_axis_values=None):
    fig = go.Figure(
        data=traces
    )

    # fig.layout.update(title_text=title, showlegend = False)
    fig.layout.update(showlegend=False, margin=dict(r=0, l=0, b=0, t=0))

    fig.update_xaxes(
        ticktext=x_axis_text,
        tickvals=x_axis_values,
        title_text=x_axis_label
    )

    if y_axis_values:
        fig.update_yaxes(
            range=y_axis_values,
        )

    fig.update_yaxes(title_text=y_axis_label)

    fig.show()


def get_min_max_avg_traces(data):

    y1 = data[0]['y']
    y2 = data[1]['y']
    y3 = data[2]['y']
    y4 = data[3]['y']
    y5 = data[4]['y']

    min_len = min(len(y1), len(y2), len(y3), len(y4), len(y5))

    y1 = y1[:min_len]
    y2 = y2[:min_len]
    y3 = y3[:min_len]
    y4 = y4[:min_len]
    y5 = y5[:min_len]

    data_stack = np.stack((y1, y2, y3, y4, y5), axis=0)

    upper_bound_data = np.max(data_stack, axis=0)
    lower_bound_data = np.min(data_stack, axis=0)
    avg_data = np.average(data_stack, axis=0)

    x_data = data[0]['x'][:min_len]

    upper_bound = go.Scatter(
        name='Upper Bound',
        x=x_data,
        y=upper_bound_data,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    trace = go.Scatter(
        name='Average',
        x=x_data,
        y=avg_data,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    lower_bound = go.Scatter(
        name='Lower Bound',
        x=x_data,
        y=lower_bound_data,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines')

    # Trace order can be important with continuous error bars
    traces = [lower_bound, trace, upper_bound]
    return traces


if __name__ == '__main__':

    with open('bert_base_cased_finetuned_squad_results.json') as json_file:
        results = json.load(json_file)
        traces = []
        for run in results:
            x_axis_values, traces_run = get_squad_run_data(results[run], run)
            traces.extend(traces_run)

    print('babau')
