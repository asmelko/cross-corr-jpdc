if ~(parallel.gpu.GPUDevice.isAvailable)
    fprintf(['\n\t**GPU not available. Stopping.**\n']);
    return;
else
    dev = gpuDevice;
    fprintf(...
    'GPU detected (%s, %d multiprocessors, Compute Capability %s)\n',...
    dev.Name, dev.MultiprocessorCount, dev.ComputeCapability);
end

fprintf(...
    'Computing cross-correlation of "%s" and "%s" using the %s algorithm, writing results to "%s"\n',...
    in1_path, in2_path, alg, out_path);

% in1_path and in2_path set externally
[in1_matrix_size, in1_num_matrices, in1_data] = parseInput(in1_path);
[in2_matrix_size, in2_num_matrices, in2_data] = parseInput(in2_path);

switch alg
case 'one_to_one'
    assert(in1_num_matrices == 1)
    assert(in2_num_matrices == 1)

    g_in1 = gpuArray(in1_data);
    g_in2 = gpuArray(in2_data);

    g_xcor = xcorr2(g_in1, g_in2);


    result_matrix = gather(g_xcor);
    header = [size(result_matrix), 1];
case 'one_to_many'
    assert(in1_num_matrices == 1)

    g_in1 = gpuArray(in1_data);
    result_matrix_size = in1_matrix_size + in2_matrix_size - 1;
    result_matrix = zeros(result_matrix_size(1) * in2_num_matrices, result_matrix_size(2));
    for t = 1:in2_num_matrices
        % disp(t)
        % disp(1 + (t-1)*in2_matrix_size(1))
        % disp(t*in2_matrix_size(1))
        % disp(1 + (t-1)*result_matrix_size(1))
        % disp(t*result_matrix_size(1))
        g_in2 = gpuArray(in2_data(1 + (t-1)*in2_matrix_size(1):t*in2_matrix_size(1),:));
        g_xcor = xcorr2(g_in1, g_in2);
        result_matrix(1 + (t-1)*result_matrix_size(1):t*result_matrix_size(1),:) = gather(g_xcor);
    end

    header = [result_matrix_size, in2_num_matrices];
case 'n_to_m'
    error('Algorithm %s not implemented', alg)
case 'n_to_nm'
    assert(mod(in2_num_matrices, in1_num_matrices) == 0)
    error('Algorithm %s not implemented', alg)
otherwise
    error('Unknown algorithm type %s', alg)
end

% out_path set externally
fid = fopen(path, 'w');
fprintf(fid, '# %u,%u,%u', header(1), header(2), header(3));
fclose(fid);
writematrix(result_matrix, out_path, "Delimiter", ",", "FileType", "text", "WriteMode", "append");

disp('Done')

function [matrix_size, num_matrices, data] = parseInput(path)
    fid = fopen(path, 'r');
    header = textscan(fid, '# %u%u%u', 'Delimiter', ',', 'ReturnOnError', 1);
    fclose(fid);
    assert(size(header, 2) >= 2);

    matrix_size = [cell2mat(header(1)), cell2mat(header(2))];
    if (size(header, 2) == 3)
        num_matrices = cell2mat(header(3));
    else
        num_matrices = 1;
    end
    data = readmatrix(path, 'NumHeaderLines', 1);
end