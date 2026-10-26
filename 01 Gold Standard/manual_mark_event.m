function manual_event = manual_mark_event(base_drive, patient_id, tm_choice, test_choice, trial_choice, vertical_axis, event_type, mark_mode)
% MANUAL_MARK_EVENT  Plots the stereophotogrammetry of a trial, zooms
% in on the first/last 15 seconds depending on the event type, and allows you to click
% to mark the start and/or end of the event using ginput.
%
% event_type: 'STS' -> zoom in on the first 15 seconds of the signal
%             'SdS' -> zoom in on the last 15 seconds of the signal
%
% mark_mode: 'both' (default) = 2 clicks, start then end
%            'start'          = 1 click, start only
%            'end'            = 1 click, end only
%
% Example usage:
%   ev = manual_mark_event('/path/to/PFF_cohort_MobD', 'XXXX', ...
%                           'TimeMeasure1', 'Test4', 'Trial1', 3, 'STS', 'start');
%

    if nargin < 8
        mark_mode = 'both';
    end
    valid_modes = {'both', 'start', 'end'};
    if ~ismember(mark_mode, valid_modes)
        error('manual_mark_event:badMode', "mark_mode must be 'both', 'start' or 'end'.");
    end
    valid_events = {'STS', 'SdS'};
    if ~ismember(event_type, valid_events)
        error('manual_mark_event:badEventType', "event_type must be 'STS' or 'SdS'.");
    end

    fs = 100.0;
    % zoom_window_s = 15;

    % --- Find mat file ---
    candidates = { ...
        fullfile(base_drive, patient_id, 'Laboratory', 'data.mat'), ...
        fullfile(base_drive, patient_id, 'Laboratory', 'Data.mat'), ...
        fullfile(base_drive, patient_id, 'Laboratory', 'data_Laboratory.mat')};
    mat_path = '';
    for i = 1:numel(candidates)
        if isfile(candidates{i})
            mat_path = candidates{i};
            break
        end
    end
    if isempty(mat_path)
        error('manual_mark_event:notFound', 'No file for ID %s', patient_id);
    end
    
    %MATLAB's  load() function transparently handles both .mat v7.3 (HDF5)
    % and earlier versions, unlike Python, which requires two
    % different paths (h5py vs. scipy.io.loadmat).

    S = load(mat_path);

    % --- Navigate to TimeMeasure -> Test -> Trial -> Standards -> Stereophoto_raw -> Mrks ---
    tm = S.data.(tm_choice);
    test = tm.(test_choice);
    trial = test.(trial_choice);
    mrks = trial.Standards.Stereophoto_raw.Mrks;

    % Search for the marker whose name contains "DYNA0"
    fn = fieldnames(mrks);
    target_field = '';
    for i = 1:numel(fn)
        if contains(upper(fn{i}), 'DYNA0')
            target_field = fn{i};
            break
        end
    end
    if isempty(target_field)
        error('manual_mark_event:noMarker', 'No marker DYNA0 found in this trial');
    end

    val = mrks.(target_field);
    if size(val, 1) == 3 && size(val, 2) > 3
        val = val'; 
    end

    t = (0:size(val,1)-1) / fs;
    y = val(:, vertical_axis);
    t_max = t(end);
    lf_gd = trial.Standards.PressureInsoles_raw.LeftFoot.NormalizedPressure;
    rf_gd = trial.Standards.PressureInsoles_raw.RightFoot.NormalizedPressure;
    p_tot = sum(lf_gd, 2) + sum(rf_gd, 2);

    % --- Plot ---
    figure('Name', sprintf('Patient %s - %s %s', patient_id, test_choice, trial_choice));
    plot(t(2059:6366), y(2059:6366), 'LineWidth', 1.3);
    % hold on
    % plot(t,p_tot)
    grid on
    xlabel('Time (s)');
    ylabel('Position (mm)');

 % Initial zoom: first 15 seconds for STS, last 15 seconds for SdS. The
    % complete signal remains on the graph, so you can always zoom
    % out (double-click / zoom toolbar) if you need more context.
    % if strcmp(event_type, 'STS')
    %     xlim([0, min(zoom_window_s, t_max)]);
    % else % 'SdS'
    %     xlim([max(0, t_max - zoom_window_s), t_max]);
    % end
   

    t_start = NaN;
    t_end = NaN;

    switch mark_mode
        case 'both'
            title(sprintf('Patient %s - %s - click start and then end', patient_id, event_type));
            fprintf('Click two points on the graph: 1) Event START  2) Event END\n');
            [x_clicked, ~] = ginput(2);
            t_start = min(x_clicked);
            t_end   = max(x_clicked);
        case 'start'
            title(sprintf('Patient %s - %s - click START event', patient_id, event_type));
            fprintf('Click a point on the graph: START event\n');
            [x_clicked, ~] = ginput(1);
            t_start = x_clicked;
        case 'end'
            title(sprintf('Patient %s - %s - click END event', patient_id, event_type));
            fprintf('Click a point on the graph: END event\n');
            [x_clicked, ~] = ginput(1);
            t_end = x_clicked;
    end

    % Visual feedback on clicked points
    hold on
    if ~isnan(t_start)
        xline(t_start, '--g', 'Start', 'LineWidth', 1.2);
    end
    if ~isnan(t_end)
        xline(t_end, '--r', 'End', 'LineWidth', 1.2);
    end
    hold off

    if ~isnan(t_start) && ~isnan(t_end)
        duration_s = t_end - t_start;
    else
        duration_s = NaN; 
    end

    manual_event = struct( ...
        'PatientID', patient_id, ...
        'Test', test_choice, ...
        'Trial', trial_choice, ...
        'EventType', event_type, ...
        'T_Start_s', t_start, ...
        'T_End_s', t_end, ...
        'Duration_s', duration_s);

    fprintf('Recorded event: Start=%.2fs  End=%.2fs  Duration=%.2fs\n', ...
        t_start, t_end, duration_s);
end