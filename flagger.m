% Initialize memory space
clear all, close all

% Start Diary Log
diary ExportLog.txt

% Get Time Stamp
START_TIME = (datetime('now'));
disp(START_TIME)

% Prompt User Selection
fprintf('=============================================================================');
fprintf('\n');
location_id = input('Please enter the name of the location you are interested in: [noh][ugn][ord] \n', 's');
fprintf('\n');
loc_cat = 0;

% Check location selection
switch location_id
    case 'ord'
        % Import file
        temp_table = readtable('1958-11-01-2018-12-31ord.csv');
        
        % Check the starting and ending date
        start_time = input('Please enter start time: [1958-11-01] \n', 's');
        end_time = input('Please enter end time: [2018-12-31] \n', 's');
        fprintf('\n');
        loc_cat = 1;
    case 'noh'
        % Import file
        temp_table = readtable('1923-01-01-2002-07-31noh.csv');
        
        % Check the starting and ending date
        start_time = input('Please enter start time: [1923-01-01] \n', 's');
        end_time = input('Please enter end time: [2002-07-31] \n', 's');
        fprintf('\n');
        loc_cat = 2;
    case 'ugn'
        % Import file
        temp_table = readtable('1989-04-21-2018-12-31ugn.csv');
        
        % Check the starting and ending date
        start_time = input('Please enter start time: [1989-04-21] \n', 's');
        end_time = input('Please enter end time: [2018-12-31] \n', 's');
        fprintf('\n');
        loc_cat = 1;
    otherwise
        % Screw it
        warning('Location does not exist! \n')
        return
end

% Check the index
[logic_start, start_idx] = ismember(start_time, temp_table{:, 2});
[logic_end,    end_idx] = ismember(end_time,   temp_table{:, 2});

% Validate the dates
if logic_start == 0
    warning('The start time does not exist, please check the date and restart the program! \n');
end

if logic_end == 0
    warning('The end time does not exist, please check the date and restart the program! \n');
end

% Now we start the processing
if (logic_start == 1) && (logic_end == 1) && (loc_cat == 1)
    
    total_days = end_idx - start_idx + 1;
    fprintf(  '•	How many total days are in the time period?               %5d', total_days)
    %fprintf('             %d', total_days);
    
    precip_over_0_5 = sum(temp_table{start_idx : end_idx, 12} > 0.5);
    fprintf('\n•	How many days received more than 0.5 inches of rain?      %5d', precip_over_0_5);
    
    precip_over_1_0 = sum(temp_table{start_idx : end_idx, 12} > 1.0);
    fprintf('\n•	How many days received more than 1.0 inches of rain?      %5d', precip_over_1_0);
    
    precip_over_2_0 = sum(temp_table{start_idx : end_idx, 12} > 2.0);
    fprintf('\n•	How many days received more than 2.0 inches of rain?      %5d', precip_over_2_0);
    
    windS_over_20 = sum(temp_table{start_idx : end_idx, 7} > 20);
    fprintf('\n•	How many days had wind speeds over 20 mph?                %5d', windS_over_20);
    
    windS_over_30 = sum(temp_table{start_idx : end_idx, 7} > 30);
    fprintf('\n•	How many days had wind speeds over 30 mph?                %5d', windS_over_30);
    
    windS_over_40 = sum(temp_table{start_idx : end_idx, 7} > 40);
    fprintf('\n•	How many days had wind speeds over 40 mph?                %5d', windS_over_40);
    
    sea_below_1000 = sum(temp_table{start_idx : end_idx, 11} < 1000);
    fprintf('\n•	How many days had sea level pressures below 1000 mb?      %5d', sea_below_1000);
    
    sea_below_995 = sum(temp_table{start_idx : end_idx, 11} < 995);
    fprintf('\n•	How many days had sea level pressures below 995 mb?       %5d', sea_below_995);
    
    sea_below_990 = sum(temp_table{start_idx : end_idx, 11} < 990);
    fprintf('\n•	How many days had sea level pressures below 990 mb?       %5d', sea_below_990);
    fprintf('\n');
    
    %fprintf('\n•	How many days had ice cover on the lake more than 50%? \n')
    
    
    
end




% End logging
diary off